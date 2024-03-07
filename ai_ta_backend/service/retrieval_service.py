import inspect
import os
import time
import traceback
from typing import Dict, List, Union

import openai
from injector import inject
from langchain import hub
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.load import dumps, loads
from langchain.schema import Document

from ai_ta_backend.database.aws import AWSStorage
from ai_ta_backend.database.sql import SQLDatabase
from ai_ta_backend.database.vector import VectorDatabase
from ai_ta_backend.service.nomic_service import NomicService
from ai_ta_backend.service.posthog_service import PosthogService
from ai_ta_backend.service.sentry_service import SentryService
from ai_ta_backend.utils_tokenization import count_tokens_and_cost

OPENAI_API_TYPE = "azure"  # "openai" or "azure"


class RetrievalService:
  """
    Contains all methods for business logic of the retrieval service.
  """

  @inject
  def __init__(self, vdb: VectorDatabase, sqlDb: SQLDatabase, aws: AWSStorage, posthog: PosthogService,
               sentry: SentryService, nomicService: NomicService):
    self.vdb = vdb
    self.sqlDb = sqlDb
    self.aws = aws
    self.sentry = sentry
    self.posthog = posthog
    self.nomicService = nomicService

    openai.api_key = os.getenv("OPENAI_API_KEY")

    self.embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),  # type:ignore
        openai_api_type=OPENAI_API_TYPE,
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),  # type:ignore
        openai_api_version=os.getenv("OPENAI_API_VERSION"),  # type:ignore
    )

    self.llm = AzureChatOpenAI(
        temperature=0,
        deployment_name=os.getenv("AZURE_OPENAI_ENGINE"),  # type:ignore
        openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),  # type:ignore
        openai_api_key=os.getenv("AZURE_OPENAI_KEY"),  # type:ignore
        openai_api_version=os.getenv("OPENAI_API_VERSION"),  # type:ignore
        openai_api_type=OPENAI_API_TYPE,
    )

  def getTopContexts(self, search_query: str, course_name: str, token_limit: int = 4_000) -> Union[List[Dict], str]:
    """Here's a summary of the work.

        /GET arguments
        course name (optional) str: A json response with TBD fields.

        Returns
        JSON: A json response with TBD fields. See main.py:getTopContexts docs.
        or
        String: An error message with traceback.
        """
    try:
      start_time_overall = time.monotonic()

      found_docs: list[Document] = self.vector_search(search_query=search_query, course_name=course_name)

      pre_prompt = "Please answer the following question. Use the context below, called your documents, only if it's helpful and don't use parts that are very irrelevant. It's good to quote from your documents directly, when you do always use Markdown footnotes for citations. Use react-markdown superscript to number the sources at the end of sentences (1, 2, 3...) and use react-markdown Footnotes to list the full document names for each number. Use ReactMarkdown aka 'react-markdown' formatting for super script citations, use semi-formal style. Feel free to say you don't know. \nHere's a few passages of the high quality documents:\n"
      # count tokens at start and end, then also count each context.
      token_counter, _ = count_tokens_and_cost(pre_prompt + "\n\nNow please respond to my query: " +
                                               search_query)  # type: ignore

      valid_docs = []
      num_tokens = 0
      for doc in found_docs:
        doc_string = f"Document: {doc.metadata['readable_filename']}{', page: ' + str(doc.metadata['pagenumber']) if doc.metadata['pagenumber'] else ''}\n{str(doc.page_content)}\n"
        num_tokens, prompt_cost = count_tokens_and_cost(doc_string)  # type: ignore

        print(
            f"tokens used/limit: {token_counter}/{token_limit}, tokens in chunk: {num_tokens}, total prompt cost (of these contexts): {prompt_cost}. 📄 File: {doc.metadata['readable_filename']}"
        )
        if token_counter + num_tokens <= token_limit:
          token_counter += num_tokens
          valid_docs.append(doc)
        else:
          # filled our token size, time to return
          break

      print(f"Total tokens used: {token_counter}. Docs used: {len(valid_docs)} of {len(found_docs)} docs retrieved")
      print(f"Course: {course_name} ||| search_query: {search_query}")
      print(f"⏰ ^^ Runtime of getTopContexts: {(time.monotonic() - start_time_overall):.2f} seconds")
      if len(valid_docs) == 0:
        return []

      self.posthog.capture(
          event_name="success_get_top_contexts_OG",
          properties={
              "user_query": search_query,
              "course_name": course_name,
              "token_limit": token_limit,
              "total_tokens_used": token_counter,
              "total_contexts_used": len(valid_docs),
              "total_unique_docs_retrieved": len(found_docs),
              "getTopContext_total_latency_sec": time.monotonic() - start_time_overall,
          },
      )

      return self.format_for_json(valid_docs)
    except Exception as e:
      # return full traceback to front end
      # err: str = f"ERROR: In /getTopContexts. Course: {course_name} ||| search_query: {search_query}\nTraceback: {traceback.extract_tb(e.__traceback__)}❌❌ Error in {inspect.currentframe().f_code.co_name}:\n{e}"  # type: ignore
      err: str = f"ERROR: In /getTopContexts. Course: {course_name} ||| search_query: {search_query}\nTraceback: {traceback.print_exc} \n{e}"  # type: ignore
      traceback.print_exc()
      print(err)
      self.sentry.capture_exception(e)
      return err

  def getAll(
      self,
      course_name: str,
  ):
    """Get all course materials based on course name.
    Args:
        course_name (as uploaded on supabase)
    Returns:
        list of dictionaries with distinct s3 path, readable_filename and course_name, url, base_url.
    """

    response = self.sqlDb.getAllMaterialsForCourse(course_name)

    data = response.data
    unique_combinations = set()
    distinct_dicts = []

    for item in data:
      combination = (item['s3_path'], item['readable_filename'], item['course_name'], item['url'], item['base_url'])
      if combination not in unique_combinations:
        unique_combinations.add(combination)
        distinct_dicts.append(item)

    return distinct_dicts

  def delete_data(self, course_name: str, s3_path: str, source_url: str):
    """Delete file from S3, Qdrant, and Supabase."""
    print(f"Deleting data for course {course_name}")
    # add delete from doc map logic here
    try:
      # Delete file from S3
      bucket_name = os.getenv('S3_BUCKET_NAME')
      if bucket_name is None:
        raise ValueError("S3_BUCKET_NAME environment variable is not set")

      identifier_key, identifier_value = ("s3_path", s3_path) if s3_path else ("url", source_url)
      print(f"Deleting {identifier_value} from S3, Qdrant, and Supabase using {identifier_key}")

      # Delete from S3
      if identifier_key == "s3_path":
        self.delete_from_s3(bucket_name, s3_path)

      # Delete from Qdrant
      self.delete_from_qdrant(identifier_key, identifier_value)

      # Delete from Nomic and Supabase
      self.delete_from_nomic_and_supabase(course_name, identifier_key, identifier_value)

      return "Success"
    except Exception as e:
      err: str = f"ERROR IN delete_data: Traceback: {traceback.extract_tb(e.__traceback__)}❌❌ Error in {inspect.currentframe().f_code.co_name}:{e}"  # type: ignore
      print(err)
      self.sentry.capture_exception(e)
      return err

  def delete_from_s3(self, bucket_name: str, s3_path: str):
    try:
      self.aws.delete_file(bucket_name, s3_path)
    except Exception as e:
      print("Error in deleting file from s3:", e)
      self.sentry.capture_exception(e)

  def delete_from_qdrant(self, identifier_key: str, identifier_value: str):
    try:
      self.vdb.delete_data(os.environ['QDRANT_COLLECTION_NAME'], identifier_key, identifier_value)
    except Exception as e:
      if "timed out" in str(e):
        # Timed out is fine. Still deletes.
        pass
      else:
        print("Error in deleting file from Qdrant:", e)
        self.sentry.capture_exception(e)

  def getTopContextsWithMQR(self,
                            search_query: str,
                            course_name: str,
                            token_limit: int = 4_000) -> Union[List[Dict], str]:
    """
    New info-retrieval pipeline that uses multi-query retrieval + filtering + reciprocal rank fusion + context padding.
    1. Generate multiple queries based on the input search query.
    2. Retrieve relevant docs for each query.
    3. Filter the relevant docs based on the user query and pass them to the rank fusion step.
    4. [CANCELED BEC POINTLESS] Rank the docs based on the relevance score.
    5. Parent-doc-retrieval: Pad just the top 5 docs with expanded context from the original document.
    """
    return 'fail'

    # try:
    #   top_n_per_query = 40  # HARD CODE TO ENSURE WE HIT THE MAX TOKENS
    #   start_time_overall = time.monotonic()
    #   mq_start_time = time.monotonic()

    #   # 1. GENERATE MULTIPLE QUERIES
    #   generate_queries = (
    #       MULTI_QUERY_PROMPT | self.llm | StrOutputParser() | (lambda x: x.split("\n")) |
    #       (lambda x: list(filter(None, x)))  # filter out non-empty strings
    #   )

    #   generated_queries = generate_queries.invoke({"original_query": search_query})
    #   print("generated_queries", generated_queries)

    #   # 2. VECTOR SEARCH FOR EACH QUERY
    #   batch_found_docs_nested: list[list[Document]] = self.batch_vector_search(search_queries=generated_queries,
    #                                                                            course_name=course_name,
    #                                                                            top_n=top_n_per_query)

    #   # 3. RANK REMAINING DOCUMENTS -- good for parent doc padding of top 5 at the end.
    #   found_docs = self.reciprocal_rank_fusion(batch_found_docs_nested)
    #   found_docs = [doc for doc, score in found_docs]
    #   print(f"Num docs after re-ranking: {len(found_docs)}")
    #   if len(found_docs) == 0:
    #     return []
    #   print(f"⏰ Total multi-query processing runtime: {(time.monotonic() - mq_start_time):.2f} seconds")

    #   # 4. FILTER DOCS
    #   filtered_docs = filter_top_contexts(contexts=found_docs, user_query=search_query, timeout=30, max_concurrency=180)
    #   if len(filtered_docs) == 0:
    #     return []

    #   # 5. TOP DOC CONTEXT PADDING // parent document retriever
    #   final_docs = context_parent_doc_padding(filtered_docs, search_query, course_name)
    #   print(f"Number of final docs after context padding: {len(final_docs)}")

    #   pre_prompt = "Please answer the following question. Use the context below, called your documents, only if it's helpful and don't use parts that are very irrelevant. It's good to quote from your documents directly, when you do always use Markdown footnotes for citations. Use react-markdown superscript to number the sources at the end of sentences (1, 2, 3...) and use react-markdown Footnotes to list the full document names for each number. Use ReactMarkdown aka 'react-markdown' formatting for super script citations, use semi-formal style. Feel free to say you don't know. \nHere's a few passages of the high quality documents:\n"
    #   token_counter, _ = count_tokens_and_cost(pre_prompt + '\n\nNow please respond to my query: ' +
    #                                            search_query)  # type: ignore

    #   valid_docs = []
    #   num_tokens = 0
    #   for doc in final_docs:
    #     doc_string = f"Document: {doc['readable_filename']}{', page: ' + str(doc['pagenumber']) if doc['pagenumber'] else ''}\n{str(doc['text'])}\n"
    #     num_tokens, prompt_cost = count_tokens_and_cost(doc_string)  # type: ignore

    #     print(f"token_counter: {token_counter}, num_tokens: {num_tokens}, max_tokens: {token_limit}")
    #     if token_counter + num_tokens <= token_limit:
    #       token_counter += num_tokens
    #       valid_docs.append(doc)
    #     else:
    #       # filled our token size, time to return
    #       break

    #   print(f"Total tokens used: {token_counter} Used {len(valid_docs)} of total unique docs {len(found_docs)}.")
    #   print(f"Course: {course_name} ||| search_query: {search_query}")
    #   print(f"⏰ ^^ Runtime of getTopContextsWithMQR: {(time.monotonic() - start_time_overall):.2f} seconds")

    #   if len(valid_docs) == 0:
    #     return []

    #   self.posthog.capture('distinct_id_of_the_user',
    #                        event='filter_top_contexts_succeeded',
    #                        properties={
    #                            'user_query': search_query,
    #                            'course_name': course_name,
    #                            'token_limit': token_limit,
    #                            'total_tokens_used': token_counter,
    #                            'total_contexts_used': len(valid_docs),
    #                            'total_unique_docs_retrieved': len(found_docs),
    #                        })

    #   return self.format_for_json_mqr(valid_docs)
    # except Exception as e:
    #   # return full traceback to front end
    #   err: str = f"ERROR: In /getTopContextsWithMQR. Course: {course_name} ||| search_query: {search_query}\nTraceback: {traceback.format_exc()}❌❌ Error in {inspect.currentframe().f_code.co_name}:\n{e}"  # type: ignore
    #   print(err)
    #   sentry_sdk.capture_exception(e)
    #   return err

  def format_for_json_mqr(self, found_docs) -> List[Dict]:
    """
    Same as format_for_json, but for the new MQR pipeline.
    """
    for found_doc in found_docs:
      if "pagenumber" not in found_doc.keys():
        print("found no pagenumber")
        found_doc['pagenumber'] = found_doc['pagenumber_or_timestamp']

    contexts = [
        {
            'text': doc['text'],
            'readable_filename': doc['readable_filename'],
            'course_name ': doc['course_name'],
            's3_path': doc['s3_path'],
            'pagenumber': doc['pagenumber'],
            'url': doc['url'],  # wouldn't this error out?
            'base_url': doc['base_url'],
        } for doc in found_docs
    ]

    return contexts

  def delete_from_nomic_and_supabase(self, course_name: str, identifier_key: str, identifier_value: str):
    try:
      response = self.sqlDb.getMaterialsForCourseAndKeyAndValue(course_name, identifier_key, identifier_value)
      data = response.data[0]  # single record fetched
      nomic_ids_to_delete = [str(data['id']) + "_" + str(i) for i in range(1, len(data['contexts']) + 1)]

      # delete from Nomic
      # check if project exists
      response = self.sqlDb.getProjectsMapForCourse(course_name)
      if response.data:
        project_id = response.data[0]['doc_map_id']
      else:
        return "No document map found for this course"
      res = self.nomicService.delete_from_document_map(project_id, nomic_ids_to_delete)

      # delete from Supabase
      self.sqlDb.deleteMaterialsForCourseAndKeyAndValue(course_name, identifier_key, identifier_value)
    except Exception as e:
      print(f"Error in deleting file from Nomic or Supabase using {identifier_key}: {identifier_value}", e)
      self.sentry.capture_exception(e)

  def vector_search(self, search_query, course_name):
    top_n = 80
    # EMBED
    openai_start_time = time.monotonic()
    print("OPENAI_API_TYPE", OPENAI_API_TYPE)
    user_query_embedding = self.embeddings.embed_query(search_query)
    openai_embedding_latency = time.monotonic() - openai_start_time

    # SEARCH
    self.posthog.capture(
        event_name="vector_search_invoked",
        properties={
            "user_query": search_query,
            "course_name": course_name,
        },
    )
    qdrant_start_time = time.monotonic()
    search_results = self.vdb.vector_search(search_query, course_name, user_query_embedding, top_n)

    found_docs: list[Document] = []
    for d in search_results:
      try:
        metadata = d.payload
        page_content = metadata["page_content"]
        del metadata["page_content"]
        if "pagenumber" not in metadata.keys() and "pagenumber_or_timestamp" in metadata.keys():  # type: ignore
          # aiding in the database migration...
          metadata["pagenumber"] = metadata["pagenumber_or_timestamp"]  # type: ignore

        found_docs.append(Document(page_content=page_content, metadata=metadata))  # type: ignore
      except Exception as e:
        print(f"Error in vector_search(), for course: `{course_name}`. Error: {e}")
        self.sentry.capture_exception(e)

    self.posthog.capture(
        event_name="vector_search_succeded",
        properties={
            "user_query": search_query,
            "course_name": course_name,
            "qdrant_latency_sec": time.monotonic() - qdrant_start_time,
            "openai_embedding_latency_sec": openai_embedding_latency,
        },
    )
    # print("found_docs", found_docs)
    return found_docs

  def format_for_json(self, found_docs: List[Document]) -> List[Dict]:
    """Formatting only.
        {'course_name': course_name, 'contexts': [{'source_name': 'Lumetta_notes', 'source_location': 'pg. 19', 'text': 'In FSM, we do this...'}, {'source_name': 'Lumetta_notes', 'source_location': 'pg. 20', 'text': 'In Assembly language, the code does that...'},]}

        Args:
            found_docs (List[Document]): _description_

        Raises:
            Exception: _description_

        Returns:
            List[Dict]: _description_
        """
    for found_doc in found_docs:
      if "pagenumber" not in found_doc.metadata.keys():
        print("found no pagenumber")
        found_doc.metadata["pagenumber"] = found_doc.metadata["pagenumber_or_timestamp"]

    contexts = [
        {
            "text": doc.page_content,
            "readable_filename": doc.metadata["readable_filename"],
            "course_name ": doc.metadata["course_name"],
            "s3_path": doc.metadata["s3_path"],
            "pagenumber": doc.metadata["pagenumber"],  # this because vector db schema is older...
            # OPTIONAL PARAMS...
            "url": doc.metadata.get("url"),  # wouldn't this error out?
            "base_url": doc.metadata.get("base_url"),
        } for doc in found_docs
    ]

    return contexts
