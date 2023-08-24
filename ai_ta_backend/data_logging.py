import os
import nomic
from nomic import atlas
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import time

nomic.login(os.getenv('NOMIC_API_KEY')) # login during start of flask app
NOMIC_MAP_NAME_PREFIX = 'Queries for '

def log_query_to_nomic(course_name: str, search_query: str) -> str:
  """
  Logs user query and retrieved contexts to Nomic. Must have more than 20 queries to get a map.

  TODO: Better handle courses that have less than 20 queries
  """

  project_name = NOMIC_MAP_NAME_PREFIX + course_name
  start_time = time.monotonic()

  embeddings_model = OpenAIEmbeddings() # type: ignore
  embeddings = np.array(embeddings_model.embed_query(search_query)).reshape(1, 1536)
  data = [{'course_name': course_name, 'query': search_query, 'id': time.time()}]

  try:
    # slow call, about 0.6 sec
    project = atlas.AtlasProject(name=project_name, add_datums_if_exists=True)
    # mostly async call (0.35 to 0.5 sec)
    project.add_embeddings(embeddings=embeddings, data=data)
  except Exception as e:
    print("Nomic map does not exist yet, probably because you have less than 20 queries on your project: ", e)
  
  print(f"⏰ Nomic logging runtime: {(time.monotonic() - start_time):.2f} seconds")
  return f"Successfully logged for {course_name}"

def get_nomic_map(course_name: str):
  """
  Returns iframe string of the Nomic map given a course name.
  """
  project_name = NOMIC_MAP_NAME_PREFIX + course_name
  start_time = time.monotonic()

  try:
    project = atlas.AtlasProject(name=project_name, add_datums_if_exists=True)
  except Exception as e:
    err = f"Nomic map does not exist yet, probably because you have less than 20 queries on your project: {e}"
    print(err)
    return err
  
  with project.wait_for_project_lock() as project:
    rebuild_start_time = time.monotonic()
    project.rebuild_maps()
    print(f"⏰ Nomic _only_ map rebuild: {(time.monotonic() - rebuild_start_time):.2f} seconds")
  map = project.get_map(project_name)
  print(f"⏰ Nomic Full Map Retrieval: {(time.monotonic() - start_time):.2f} seconds")
  return map._iframe()
