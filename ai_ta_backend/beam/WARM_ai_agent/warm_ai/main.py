import os
import beam
from dotenv import load_dotenv
from typing import Any, Dict, List
from ai_ta_backend.beam.WARM_ai_agent.warm_ai.agents.sql_agent import SQLAIAgent
from ai_ta_backend.beam.WARM_ai_agent.warm_ai.utils.sql import extract_sql_query

if beam.env.is_remote():
    import os
    from dotenv import load_dotenv
    from typing import Any, Dict, List
    from ai_ta_backend.beam.WARM_ai_agent.warm_ai.agents.sql_agent import SQLAIAgent
    from ai_ta_backend.beam.WARM_ai_agent.warm_ai.utils.sql import extract_sql_query

requirements = [
    'langchain_community>=0.0.1',
    'SQLAlchemy>=2.0.0',
    'langchain_core>=0.0.1',
    'langchain_openai>=0.0.1',
    'langgraph>=0.0.1',
    'pydantic>=2.0.0',
    'python-dotenv>=0.0.1',
    'pyodbc>=4.0.0',
    'packaging>=23.2',  # Explicitly set packaging version to resolve conflict
    'fastapi>=0.109.0',  # Remove fixed version
    'gunicorn>=20.1.0',  # Remove fixed version
    'uvicorn>=0.29.0',  # Remove fixed version
    'requests>=2.28.1',  # Remove fixed version
    'MarkupSafe>=2.0.0',  # Remove fixed version
    'websockets>=11.0.3'  # Remove fixed version
]

volume_path = "./models"

image = beam.Image(
    python_version="python3.10",
    python_packages=requirements,
    commands=["apt-get update && apt-get install unixodbc-dev -y"],
)

ourSecrets = [
    "DB_DRIVER",
    "DB_SERVER",
    "DB_NAME",
    "DB_TRUSTED_CONNECTION",
    "DB_TRUST_SERVER_CERT",
    "OPENAI_API_KEY",
]


# def loader():
#   """Model weights are downloaded once at image build time using the @build hook and saved into the image. 'Baking' models into the modal.Image at build time provided the fastest cold start. """
#   print("Inside loader")

#   model_url = "https://assets.kastan.ai/pest_detection_model_weights.pt"
#   model_path = f'{volume_path}/pest_detection_model_weights.pt'
#   start_time = time.monotonic()

#   print("After model path: ", model_path)

#   # save/load model to/from volume
#   if not os.path.exists(model_path):
#     print("Downloading model from the internet")
#     response = requests.get(model_url, timeout=60)
#     with open(model_path, "wb") as f:
#       f.write(response.content)
#     model = YOLO(response.content)
#   else:
#     print("Loading model from volume")
#     model = YOLO(model_path)

#   s3 = boto3.client(
#       service_name="s3",
#       endpoint_url=f"https://{os.environ['CLOUDFLARE_ACCOUNT_ID']}.r2.cloudflarestorage.com",
#       aws_access_key_id=os.environ['CLOUDFLARE_ACCESS_KEY_ID'],
#       aws_secret_access_key=os.environ['CLOUDFLARE_SECRET_ACCESS_KEY'],
#       # region_name="<location>", # Must be one of: wnam, enam, weur, eeur, apac, auto
#   )

#   print(f"⏰ Runtime to load model: {(time.monotonic() - start_time):.2f} seconds")
#   return model, s3


# autoscaler = QueueDepthAutoscaler(max_tasks_per_replica=2, max_replicas=3)
@beam.endpoint(name="Microsoft SQL AI Agent WARM",
               workers=1,
               cpu=1,
               memory="3Gi",
               max_pending_tasks=10,
               timeout=60,
               keep_warm_seconds=0,
               secrets=ourSecrets,
            #    on_start=loader,
               image=image,
            #    volumes=[beam.Volume(name="my_models", mount_path=volume_path)])
)
def main(context, **inputs: Dict[str, Any]):
    print("Starting WARM AI Agent...")

    # Load environment variables
    load_dotenv()
    
    # Build connection string from environment variables
    connection_str = (
        f"Driver={os.getenv('DB_DRIVER')};"
        f"Server={os.getenv('DB_SERVER')};"
        f"Database={os.getenv('DB_NAME')};"
        f"Trusted_Connection={os.getenv('DB_TRUSTED_CONNECTION')};"
        f"TrustServerCertificate={os.getenv('DB_TRUST_SERVER_CERT')};"
    )

    print("Initializing agent with connection string...")
    agent = SQLAIAgent(connection_str, os.getenv('OPENAI_API_KEY'))
    print("Connecting to database...")
    agent.connect()

    while True:
        # Get user input
        user_query = input("\nEnter your question (or 'quit' to exit): ")
        
        # Check for exit condition
        if user_query.lower() in ['quit', 'exit', 'q']:
            break

        print("\nProcessing natural language query...")
        results = agent.query(user_query)

        if results:
            print("\nProposed response:")
            print(results)
            
            # Extract SQL query if present
            sql_query = extract_sql_query(results)
            if sql_query:
                verify = input("\nWould you like to execute this SQL query? (y/n): ")
                if verify.lower() in ['y', 'yes']:
                    print("\nExecuting query...")
                    results = agent.execute_sql(sql_query)
                    if results:
                        print("\nQuery Results:")
                        for row in results:
                            print(row)

            # Allow user to verify and continue
            verify = input("\nWould you like to ask another question? (y/n): ")
            if verify.lower() not in ['y', 'yes']:
                break

if __name__ == "__main__":
    main()