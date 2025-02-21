"""
Metadata extraction service for Cedar Bluff documents.
"""

import asyncio
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, List

import pandas as pd
import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from trustcall import create_extractor

from ai_ta_backend.database.sql import SQLDatabase

# from utils.logging_config import setup_detailed_logging

# logger = setup_detailed_logging()


class DocumentMetadata(BaseModel):
  """
    A generic nested structure for hierarchical metadata where:
    - First level represents the primary/parent entities
    - Second level represents logical groupings/categories
    - Third+ levels represent nested attributes and their values
    The structure is flexible and can accommodate any number of nesting levels.
    """

  data: Dict[str, Dict[str,
                       Any]] = Field(description="Nested structure: parent_entity -> category -> nested_attributes")

  class Config:
    extra = "allow"
    json_schema_extra = {
        "example": {
            "data": {
                "parent_entity_1": {  # Top-level entity
                    "category_1": {  # Logical grouping
                        "attribute_1": "value_1",
                        "nested_group": {  # Can have deeper nesting
                            "sub_attribute": "value_2"
                        },
                    },
                    "category_2": {"attribute_2": "value_3"},
                }
            }
        }
    }


class DocumentMetadataProcessor:

  def __init__(self):
    self.sql_db = SQLDatabase()

    # fetch OpenAI API key from frontend
    url = os.getenv("FRONTEND_MODELS_URL")
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"projectName": "cedar-bluff"})
    api_response = requests.post(url=url, headers=headers, data=payload).json()
    encoded_openai_key = api_response["OpenAI"]["apiKey"]

    signing_key = os.getenv("NEXT_PUBLIC_SIGNING_KEY")
    openai_key = self.decrypt(encrypted_text=encoded_openai_key, key=signing_key)

    self.client = ChatOpenAI(api_key=openai_key, model="gpt-4o", temperature=0)
    self.extractor = create_extractor(self.client, tools=[DocumentMetadata], tool_choice="DocumentMetadata")

  def decrypt(self, encrypted_text: str, key: str) -> str:
    if not encrypted_text or not key:
      raise ValueError("Encrypted text or key is missing")

    parts = encrypted_text.split(".")
    if len(parts) != 3:
      raise ValueError("Invalid encrypted text format")

    version, encrypted_base64, iv_base64 = parts

    if version != "v1":
      raise ValueError(f"Unsupported encryption version: {version}")

    # Convert base64-encoded data back to bytes
    encrypted_bytes = base64.b64decode(encrypted_base64)
    iv = base64.b64decode(iv_base64)

    # Extract authentication tag (last 16 bytes of encrypted data)
    if len(encrypted_bytes) < 16:
      raise ValueError("Invalid encrypted data: Too short to contain tag")

    ciphertext, tag = encrypted_bytes[:-16], encrypted_bytes[-16:]

    # Hash the key using SHA-256
    key_hash = hashlib.sha256(key.encode()).digest()

    # Setup AES-GCM decryption with IV and authentication tag
    cipher = Cipher(algorithms.AES(key_hash), modes.GCM(iv, tag), backend=default_backend())
    decryptor = cipher.decryptor()

    try:
      decrypted_bytes = decryptor.update(ciphertext) + decryptor.finalize()
      return decrypted_bytes.decode()
    except Exception as e:
      raise ValueError("Failed to decrypt data: " + str(e))

  def process_documents(self, input_prompt: str, document_ids: List):
    """
    This function generates metadata from the text chunks and table JSONs extracted from the PDFs.
    """
    try:
      curr_run_id = self.sql_db.getLastRunID().data[0]["run_id"] + 1
      print(f"Current run ID: {curr_run_id}")
      yield {"run_id": curr_run_id, "status": "started"}

      async def process_batch(batch, existing_metadata, user_prompt):
        print(f"Processing batch...")
        output_format = """
                        {
                            "data": {
                                "<parent_entity>": {            # Primary/top-level entity identifier
                                    "<category>": {             # Logical grouping of related information
                                        "<attribute>": <value>   # Attribute-value pairs, can be nested further
                                    }
                                }
                            }
                        }
                """
        prompt = """You are an expert at analyzing documents and extracting structured metadata into clean, hierarchical formats.
                    Your task is to extract and organize all information into a nested structure that preserves relationships and values. 
                    Carefully structure the output to best respond to the user's request. 
                    Since information will be shared iteratively and your responses will be saved in existing metadata, feel free to restructure the output so that it is most useful to the user. 
                    This might be the case especially when there's lower level information that is not shared yet.

                    Important Instructions:
                        1. PARENT ENTITY IDENTIFICATION
                          - Identify the main top-level entities that serve as primary identifiers
                          - These group related information together at the highest level

                        2. CATEGORY IDENTIFICATION
                          - Identify distinct logical groups of information
                          - Each category should group related attributes together

                        3. VALUE MAPPING
                          - For each parent entity and category:
                            * Map every attribute to its exact value
                            * Keep text exactly as shown
                            * Preserve all formatting and units
                            * Maintain relationships between values

                        4. DATA QUALITY
                          - Preserve exact text and values
                          - Don't modify or summarize values
                          - Maintain all relationships and hierarchies
                          - Don't skip any information

                        Remember:
                        - Focus on preserving relationships and hierarchies
                        - Be thorough in extracting all relevant data
                        - Keep the structure consistent across all entries
                        - Allow for flexible nesting where needed

                        Critical Requirements:
                        - Preserve exact text and values
                        - Include all fields and values
                        - Maintain hierarchical relationships
                        - Keep formatting and units intact
                        - Don't skip any information
                        - Don't summarize or modify values
                        - Preserve all specifications, details, attributes, and relationships
                        - Do not make up any information, only extract what is present in the document or the existing metadata

                    Output Format: 
                """

        try:
          prompt = (prompt + output_format
                   )  # adding this way because including it in the main prompt gives formatting errors.
          result = await self.extractor.ainvoke({
              "messages": [
                  {
                      "role": "system",
                      "content": prompt
                  },
                  {
                      "role": "user",
                      "content": user_prompt + "\n\n Document Chunks: ".join(batch),
                  },
              ],
              "existing": ({
                  "DocumentMetadata": existing_metadata
              } if existing_metadata else None),
          })

          if result:
            metadata = DocumentMetadata(**result["responses"][0].model_dump())
            return metadata
        except Exception as e:
          print("Error processing batch:", e)
          return None

      # Process each document's chunks
      print(f"Total documents: {len(document_ids)}")
      for document_id in document_ids:
        try:
          # insert run status as started
          self.sql_db.insertCedarRun({"run_id": curr_run_id, "document_id": document_id, "run_status": "in_progress", "prompt": input_prompt})
          # Fetch document chunks
          doc_chunks = self.sql_db.getCedarChunks(document_id).data
          if not doc_chunks:
            print(f"No chunks found for document ID: {document_id}")
            continue

          print(f"Processing document ID: {document_id}")

          table_chunks = [chunk for chunk in doc_chunks if chunk["chunk_type"] in ["Table", "TableChunk"]]
          content_chunks = [chunk for chunk in doc_chunks if chunk["chunk_type"] not in ["Table", "TableChunk"]]

          print(f"Table chunks: {len(table_chunks)}")
          print(f"Content chunks: {len(content_chunks)}")

          content_texts = [chunk["content"] for chunk in content_chunks]
          batch_size = 10
          existing_metadata = None
          has_successful_batch = False

          chunk_batches = [[(json.dumps(chunk["table_data"]) if chunk["table_data"] else chunk["table_html"])]
                           for chunk in table_chunks
                          ] + [content_texts[i:i + batch_size] for i in range(0, len(content_texts), batch_size)]
          print(f"Total chunk batches: {len(chunk_batches)}")

          for batch_idx, batch in enumerate(chunk_batches):
            if not any(batch):
              continue

            # Run the async function in a new event loop
            metadata = asyncio.run(process_batch(batch, existing_metadata, input_prompt))

            if metadata:
              existing_metadata = metadata.model_dump()
              print(f"Length of metadata: {len(metadata.data)}")

              # Save metadata for this batch
              for parent_entity, entity_data in metadata.data.items():
                print(f"Parent entity: {parent_entity}")
                doc_metadata_row = {
                    "document_id": document_id,
                    "run_id": curr_run_id,
                    "field_name": parent_entity,
                    "field_value": entity_data,
                    "confidence_score": 90,
                    "extraction_method": "gpt-4o",
                }
                self.sql_db.insertCedarDocumentMetadata(doc_metadata_row)
              has_successful_batch = True
              print(f"Document ID {document_id}, Batch {batch_idx + 1}: metadata saved!")

          # Update document status based on overall success
          if has_successful_batch:
            self.sql_db.updateCedarRunStatus(doc_id=document_id, run_id=curr_run_id, data={"run_status": "completed"})
          else:
            self.sql_db.updateCedarRunStatus(
                doc_id=document_id,
                run_id=curr_run_id,
                data={
                    "run_status": "failed",
                    "last_error": "No metadata extracted from any batch in the document",
                },
            )

        except Exception as e:
          print("Error in doc level metadata extraction: ", e)
          self.sql_db.updateCedarRunStatus(doc_id=document_id,
                                           run_id=curr_run_id,
                                           data={
                                               "run_status": "failed",
                                               "last_error": str(e)
                                           })
          continue

      yield {"run_id": curr_run_id, "status": "completed"}
    except Exception as e:
      print("Error: ", str(e))
      yield {"run_id": curr_run_id, "status": "failed", "error": str(e)}

  def download_metadata_csv(self, run_ids: List[int]) -> List[str]:
    """
        This function downloads the metadata from the database and saves it as a CSV file.
    """
    try:
      # fetch all processed docs
      limit = 100
      offset = 0
      runs_ids_str = ",".join(map(str, run_ids))
      print(f"Run IDs: {runs_ids_str}")
      metadata = []
      while True:
        data = self.sql_db.getRunData(runs_ids_str, limit, offset).data
        if not data:
          break
        metadata.extend(data)
        offset += limit

      final_metadata = []
      for doc in metadata:

        final_metadata.append({
            "run_id": doc["run_id"],
            "document_id": doc["document_id"],
            "readable_filename": doc["readable_filename"],
            "field_name": doc["field_name"],
            "field_value": json.dumps(doc["field_value"]),
        })

      print(f"Final metadata: {len(final_metadata)}")
      # Save metadata as CSV
      if len(final_metadata) > 0:
        df = pd.DataFrame(final_metadata)
        csv_file = "metadata.csv"
        file_path = os.path.join(os.getcwd(), csv_file)
        df.to_csv(file_path, index=False, encoding="utf-8")
        return [file_path, csv_file, os.getcwd()]

      return []
    except Exception as e:
      print("Error: ", e)
      return []
