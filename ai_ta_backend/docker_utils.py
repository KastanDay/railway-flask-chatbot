from pathlib import Path
import docker
import os

from supabase.client import create_client
from docker.errors import NotFound, BuildError, ContainerError, ImageNotFound, APIError
import traceback
import logging
from newrelic_telemetry_sdk import Log, LogClient

# Initialize Supabase client
supabase_url = os.environ["SUPABASE_URL"]
supabase_key = os.environ["SUPABASE_API_KEY"]
supabase = create_client(supabase_url, supabase_key)
    
logging.basicConfig(level=logging.INFO)
logging.getLogger('docker').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)

# Initialize New Relic Client
log_client = LogClient(os.environ['NEW_RELIC_LICENSE_KEY'])

def get_docker_client():
  # Initialize Docker client
  try:
    docker_client = docker.from_env()
    return docker_client
  except Exception as e:
    print(f"ERROR: Docker client must be running -- Error while setting up docker client: {e}")
    print(traceback.print_exc())
    raise e


def build_docker_image(image_name):
  """Build a Docker image with the given name.

  Args:
      image_name (str): The Docker image name.
  """
  docker_client = get_docker_client()

  dockerfile_path = str(Path(os.getcwd() + "/ai_ta_backend/agents"))
  print(f"Building docker image: {image_name}")
  print("Current working directory: ", os.getcwd())
  # dockerfile_path = "ai_ta_backend/agents"
  try:
    img, logs = docker_client.images.build(path=dockerfile_path, tag=image_name, quiet=False) #type:ignore
    for log in logs:
      if 'stream' in log:
        print(f"Build logs: {log['stream'].strip()}")
      elif 'error' in log:
        print(f"Error: {log['error']}")
    return img
  except BuildError as e:
    print(f"Error msg: {e.msg}")
    for log in e.build_log:
      if 'stream' in log:
        print(f"Build logs on error: {log['stream'].strip()}")
      elif 'error' in log:
        print(f"Error log: {log['error']}")
    print(traceback.print_exc())
  except Exception as e:
    print(f"Error: {e}")
    print(traceback.print_exc())
  return None


def run_docker_container(image_name, command, volume_name):
  """Run a Docker container with the given image name and command.

  Args:
      image_name (str): The Docker image name.
      command (str): The command to run in the Docker container.
      volume_name (str): The name of the Docker volume.
  """
  docker_client = get_docker_client()

  try:
    volume = docker_client.volumes.get(volume_name)
  except NotFound:
    # If the volume does not exist, create it
    volume = docker_client.volumes.create(name=volume_name)

  image_name_without_tag = image_name.split(':')[0].split('/')[-1]

  # ! TODO Figure out how to RESUME a container if it already exists
  try:
    # Check if the container exists
    #! BAD: This doesn't use the latest code... it uses the code from when the container was created
    container = docker_client.containers.get(image_name_without_tag)
    print("🎉 Container exists, resuming...")
    # If the container exists, start it
    container.start()
  except NotFound:
    print("🤯 Container does not exist, creating a new one...")
    # If the container does not exist, run a new one
    container = docker_client.containers.run(
        image_name, 
        command, 
        name=image_name_without_tag, 
        detach=True, 
        volumes={volume_name: {'bind': '/volume', 'mode': 'rw'}}
    )
  except Exception as e:
    print(f"Error: {e}")
    print(traceback.print_exc())


def check_and_insert_image_name(image_name, langsmith_run_id):
  """Check if the image name exists in the Supabase table, if not, insert it and build a Docker image.

  Args:
      image_name (str): The Docker image name.
  """
  docker_images = []
  img = None
  result = supabase.table("docker_images").select("image_name").eq("image_name", image_name).execute()
  if not result.data:
    # If the image name does not exist in the table, insert it and build a Docker image
    supabase.table("docker_images").insert({"image_name": image_name, "langsmith_id": langsmith_run_id}).execute()

  # Always build the Docker image (to ensure it's up to date)
  return build_docker_image(image_name)
