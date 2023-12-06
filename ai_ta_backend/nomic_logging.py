import datetime
import inspect
import os
import time
import traceback
import requests
import nomic
import numpy as np
import pandas as pd
import supabase
from langchain.embeddings import OpenAIEmbeddings
from nomic import AtlasProject, atlas
from nomic.project import AtlasClass


def log_convo_to_nomic(course_name: str, conversation) -> str:
  nomic.login(os.getenv('NOMIC_API_KEY'))  # login during start of flask app
  NOMIC_MAP_NAME_PREFIX = 'Conversation Map for '

  """
  Logs conversation to Nomic.
  1. Check if map exists for given course
  2. Check if conversation ID exists 
    - if yes, delete and add new data point
    - if no, add new data point
  3. Keep current logic for map doesn't exist - update metadata
  """
  print(f"in log_convo_to_nomic() for course: {course_name}")

  messages = conversation['conversation']['messages']
  user_email = conversation['conversation']['user_email']
  conversation_id = conversation['conversation']['id']

  # we have to upload whole conversations
  # check what the fetched data looks like - pandas df or pyarrow table
  # check if conversation ID exists in Nomic, if yes fetch all data from it and delete it.
  # will have current QA and historical QA from Nomic, append new data and add_embeddings()

  project_name = NOMIC_MAP_NAME_PREFIX + course_name
  start_time = time.monotonic()
  emoji = ""

  try:
    # fetch project metadata and embbeddings
    project = AtlasProject(name=project_name, add_datums_if_exists=True)
    map_metadata_df = project.maps[1].data.df # type: ignore
    map_embeddings_df = project.maps[1].embeddings.latent
    map_metadata_df['id'] = map_metadata_df['id'].astype(int)
    last_id = map_metadata_df['id'].max()

    if conversation_id in map_metadata_df.values:
      # store that convo metadata locally
      prev_data = map_metadata_df[map_metadata_df['conversation_id'] == conversation_id]
      prev_index = prev_data.index.values[0]
      embeddings = map_embeddings_df[prev_index - 1].reshape(1, 1536)
      prev_convo = prev_data['conversation'].values[0]
      prev_id = prev_data['id'].values[0]
      created_at = pd.to_datetime(prev_data['created_at'].values[0]).strftime('%Y-%m-%d %H:%M:%S')

      # delete that convo data point from Nomic, and print result
      print("Deleting point from nomic:", project.delete_data([str(prev_id)]))

      # prep for new point
      first_message = prev_convo.split("\n")[1].split(": ")[1]

      # select the last 2 messages and append new convo to prev convo
      messages_to_be_logged = messages[-2:]
      for message in messages_to_be_logged:
        if message['role'] == 'user':
          emoji = "🙋 "
        else:
          emoji = "🤖 "

        prev_convo += "\n>>> " + emoji + message['role'] + ": " + message['content'] + "\n"

      # modified timestamp
      current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      # update metadata
      metadata = [{
          "course": course_name,
          "conversation": prev_convo,
          "conversation_id": conversation_id,
          "id": last_id + 1,
          "user_email": user_email,
          "first_query": first_message,
          "created_at": created_at,
          "modified_at": current_time
      }]
    else:
      print("conversation_id does not exist")

      # add new data point
      user_queries = []
      conversation_string = ""
      first_message = messages[0]['content']
      user_queries.append(first_message)

      for message in messages:
        if message['role'] == 'user':
          emoji = "🙋 "
        else:
          emoji = "🤖 "
        conversation_string += "\n>>> " + emoji + message['role'] + ": " + message['content'] + "\n"

      # modified timestamp
      current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      metadata = [{
          "course": course_name,
          "conversation": conversation_string,
          "conversation_id": conversation_id,
          "id": last_id + 1,
          "user_email": user_email,
          "first_query": first_message,
          "created_at": current_time,
          "modified_at": current_time
      }]

      # create embeddings
      embeddings_model = OpenAIEmbeddings() # type: ignore
      embeddings = embeddings_model.embed_documents(user_queries)

    # add embeddings to the project
    project = atlas.AtlasProject(name=project_name, add_datums_if_exists=True)
    with project.wait_for_project_lock():
      project.add_embeddings(embeddings=np.array(embeddings), data=pd.DataFrame(metadata))
      project.rebuild_maps()

    print(f"⏰ Nomic logging runtime: {(time.monotonic() - start_time):.2f} seconds")
    return f"Successfully logged for {course_name}"

  except Exception as e:
    # Error handling - the below error is for when the project does not exist
    if str(e) == 'You must specify a unique_id_field when creating a new project.':
      # project does not exist, so create it
      result = create_nomic_map(course_name, conversation)
      if result is None:
        print("Nomic map does not exist yet, probably because you have less than 20 queries on your project: ", e)
        return f"Logging failed for {course_name}"
      else:
        print(f"⏰ Nomic logging runtime: {(time.monotonic() - start_time):.2f} seconds")
        return f"Successfully logged for {course_name}"
    else:
      # for rest of the errors - return fail
      err = f"❌❌ Error in log_convo_to_nomic() for course `{course_name}`: {e}\nTraceback: {traceback.extract_tb(e.__traceback__)}"  # type: ignore
      print(err)
      return err
    
    
def get_nomic_map(course_name: str):
  """
  Returns the variables necessary to construct an iframe of the Nomic map given a course name.
  We just need the ID and URL.
  Example values:
    map link: https://atlas.nomic.ai/map/ed222613-97d9-46a9-8755-12bbc8a06e3a/f4967ad7-ff37-4098-ad06-7e1e1a93dd93
    map id: f4967ad7-ff37-4098-ad06-7e1e1a93dd93
  """
  nomic.login(os.getenv('NOMIC_API_KEY'))  # login during start of flask app
  NOMIC_MAP_NAME_PREFIX = 'Conversation Map for '

  project_name = NOMIC_MAP_NAME_PREFIX + course_name
  start_time = time.monotonic()

  try:
    project = atlas.AtlasProject(name=project_name, add_datums_if_exists=True)
  except Exception as e:
    err = f"Nomic map does not exist yet, probably because you have less than 20 queries on your project: {e}"
    print(err)
    return {"map_id": None, "map_link": None}

  map = project.get_map(project_name)

  print(f"⏰ Nomic Full Map Retrieval: {(time.monotonic() - start_time):.2f} seconds")

  return {"map_id": f"iframe{map.id}", "map_link": map.map_link}


def create_nomic_map(course_name: str, log_data: list):
  """
  Creates a Nomic map for new courses and those which previously had < 20 queries.
  1. fetches supabase conversations for course
  2. appends current embeddings and metadata to it
  2. creates map if there are at least 20 queries
  """
  nomic.login(os.getenv('NOMIC_API_KEY'))  # login during start of flask app
  NOMIC_MAP_NAME_PREFIX = 'Conversation Map for '
  
  print(f"in create_nomic_map() for {course_name}")
  # initialize supabase
  supabase_client = supabase.create_client(  # type: ignore
      supabase_url=os.getenv('SUPABASE_URL'),  # type: ignore
      supabase_key=os.getenv('SUPABASE_API_KEY'))  # type: ignore

  # fetch all conversations with this new course (we expect <=20 conversations, because otherwise the map should be made already)
  response = supabase_client.table("llm-convo-monitor").select("*").eq("course_name", course_name).execute()
  data = response.data
  df = pd.DataFrame(data)

  if len(data) < 19:
    return None
  else:
    # get all queries for course and create metadata
    user_queries = []
    metadata = []
    i = 1
    conversation_exists = False

    # current log details
    log_messages = log_data['conversation']['messages'] # type: ignore
    log_user_email = log_data['conversation']['user_email'] # type: ignore
    log_conversation_id = log_data['conversation']['id'] # type: ignore

    for index, row in df.iterrows():
      user_email = row['user_email']
      created_at = pd.to_datetime(row['created_at']).strftime('%Y-%m-%d %H:%M:%S')
      convo = row['convo']
      messages = convo['messages']
      first_message = messages[0]['content']
      user_queries.append(first_message)

      # create metadata for multi-turn conversation
      conversation = ""
      if message['role'] == 'user': # type: ignore
        emoji = "🙋 "
      else:
        emoji = "🤖 "
      for message in messages:
        # string of role: content, role: content, ...
        conversation += "\n>>> " + emoji + message['role'] + ": " + message['content'] + "\n"

      # append current chat to previous chat if convo already exists
      if convo['id'] == log_conversation_id:
        conversation_exists = True
        if m['role'] == 'user': # type: ignore
          emoji = "🙋 "
        else:
          emoji = "🤖 "
        for m in log_messages:
          conversation += "\n>>> " + emoji + m['role'] + ": " + m['content'] + "\n"

      # adding modified timestamp
      current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      # add to metadata
      metadata_row = {
          "course": row['course_name'],
          "conversation": conversation,
          "conversation_id": convo['id'],
          "id": i,
          "user_email": user_email,
          "first_query": first_message,
          "created_at": created_at,
          "modified_at": current_time
      }
      metadata.append(metadata_row)
      i += 1

    # add current log as a new data point if convo doesn't exist
    if not conversation_exists:
      user_queries.append(log_messages[0]['content'])
      conversation = ""
      for message in log_messages:
        if message['role'] == 'user':
          emoji = "🙋 "
        else:
          emoji = "🤖 "
        conversation += "\n>>> " + emoji + message['role'] + ": " + message['content'] + "\n"

      # adding timestamp
      current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

      metadata_row = {
          "course": course_name,
          "conversation": conversation,
          "conversation_id": log_conversation_id,
          "id": i,
          "user_email": log_user_email,
          "first_query": log_messages[0]['content'],
          "created_at": current_time,
          "modified_at": current_time
      }
      metadata.append(metadata_row)

    metadata = pd.DataFrame(metadata)
    embeddings_model = OpenAIEmbeddings()  # type: ignore
    embeddings = embeddings_model.embed_documents(user_queries)

    # create Atlas project
    project_name = NOMIC_MAP_NAME_PREFIX + course_name
    index_name = course_name + "_convo_index"
    project = atlas.map_embeddings(
        embeddings=np.array(embeddings),
        data=metadata,  # type: ignore -- this is actually the correc type, the function signature from Nomic is incomplete
        id_field='id',
        build_topic_model=True,
        topic_label_field='first_query',
        name=project_name,
        colorable_fields=['conversation_id', 'first_query'])
    project.create_index(index_name, build_topic_model=True)
    return f"Successfully created Nomic map for {course_name}"

def list_projects(organization_id=None):
    """
    Lists all projects in an organization.
    If called without an organization id, it will list all projects in the
    current user's main organization.
    """
    c = AtlasClass()
    if organization_id is None:
        organization = c._get_current_users_main_organization()
        if organization is None:
            raise ValueError(
                "No organization id provided and no main organization found."
            )
        organization_id = organization['organization_id']
    response = requests.get(
        c.atlas_api_path + f"/v1/organization/{organization_id}",
        headers=c.header
    )
    proj_info = response.json()['projects']
    return [
        {'name': p['project_name'],
            'id': p['id'],
            'created_timestamp': p['created_timestamp']}
        for p in proj_info
    ]

def create_map_for_all_courses():
  """
  Creates a map for all the courses across UIUC.Chat
  """
  print("in create_map_for_all_courses()")
  nomic.login(os.getenv('NOMIC_API_KEY'))  # login during start of flask app
  NOMIC_MAP_NAME_PREFIX = 'Conversation Map for '
  
  # initialize supabase
  supabase_client = supabase.create_client(  # type: ignore
      supabase_url=os.getenv('SUPABASE_URL'),  # type: ignore
      supabase_key=os.getenv('SUPABASE_API_KEY'))  # type: ignore
  
  # get a list of all courses
  courses = supabase_client.table("llm-convo-monitor").select("course_name").execute()
  data = courses.data
  df = pd.DataFrame(data)
  course_list = df['course_name'].unique()
  course_list = list(course_list)
  course_list.remove(None) # there is one course called None, remove it

  project_exists = False
  existing_projects = list_projects()
  
  for course in course_list:
    try:
        # check if course already has a map
        project_name = NOMIC_MAP_NAME_PREFIX + course
        print("Project name: ", project_name)
        for project in existing_projects:
          if "name" in project and project['name'] == project_name:
            project_exists = True
            break
        
        if project_exists:
          print("Map already exists for course: ", course)
          project_exists = False
          continue
        else:
          print("Creating map for course: ", course)

          # fetch IDs of all conversations for this course
          id_response = supabase_client.table("llm-convo-monitor").select("id").eq('course_name', course).order('id', desc=False).execute()
          id_data = id_response.data
          print("Total IDs: ", len(id_data))

          if len(id_data) < 20: # cannot create a map with less than 20 queries
            continue
          else:
            course_df = pd.DataFrame()
            first_id = id_data[0]['id']
            last_id = id_data[-1]['id']

            i = 0
            current_data = []
            while len(current_data) < len(id_data):
              if len(current_data) == 0:
                response = supabase_client.table("llm-convo-monitor").select("*").eq('course_name', course).gte('id', first_id).order('id', desc=False).limit(25).execute()
              else:
                response = supabase_client.table("llm-convo-monitor").select("*").eq('course_name', course).gt('id', first_id).order('id', desc=False).limit(25).execute()
                    
              current_data += response.data
              first_id = current_data[-1]['id']
              i += 1
                
            course_df = pd.DataFrame(current_data)
            print("Number of converstions: ", len(course_df))

            user_queries = []
            metadata = []
            i = 1

            # log conversation instead of individual messages
            for index, row in course_df.iterrows():
              user_email = row['user_email']
              created_at = pd.to_datetime(row['created_at']).strftime('%Y-%m-%d %H:%M:%S')
              convo = row['convo']
              messages = convo['messages']
              first_message = messages[0]['content']
              current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              user_queries.append(first_message)
              # create metadata for multi-turn conversation
              conversation = ""
              for message in messages:
                # string of role: content, role: content, ...
                if message['role'] == 'user':
                  emoji = "🙋"
                else:
                  emoji = "🤖"
                conversation += "\n>>> " + emoji + message['role'] + ": " + message['content'] + "\n"
                # add to metadata
                metadata_row = {"course": row['course_name'], "conversation": conversation, "conversation_id": convo['id'], 
                  "id": i, "user_email": user_email, "first_query": first_message, "created_at": created_at, "modified_at": current_time}
              
              metadata.append(metadata_row)
              i += 1

              print("Length of user queries: ", len(user_queries))
              print("Length of metadata: ", len(metadata))

              metadata = pd.DataFrame(metadata)
              embeddings_model = OpenAIEmbeddings()  # type: ignore
              embeddings = embeddings_model.embed_documents(user_queries)
              embeddings = np.array(embeddings)
              print(embeddings.shape)

              # create an Atlas project
              index_name = course + "_convo_index"
              project = atlas.map_embeddings(embeddings=np.array(embeddings),
                          data=metadata,
                          id_field='id',
                          build_topic_model=True,
                          topic_label_field='first_query',
                          name=project_name,
                          colorable_fields=['conversation_id', 'first_query', 'created_at', 'modified_at'])
              print("Project Maps: ", project.maps)
              project.create_index(index_name, build_topic_model=True)
              return "Maps created for all courses!"
    except Exception as e:
      err = f"❌❌ Failed to create map create_map_for_all_courses(): {e}\nTraceback: {traceback.extract_tb(e.__traceback__)}"  # type: ignore
      print(err)
      return err

if __name__ == '__main__':
  pass
