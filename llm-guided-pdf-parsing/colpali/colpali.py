from qdrant_client import QdrantClient
from dotenv import load_dotenv
import stamina
from minio import Minio
from pdf_to_images import pdf_to_images
from qdrant_client.http import models
import os
import torch
import tqdm

import torch
from PIL import Image

# from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.models import ColQwen2, ColQwen2Processor

load_dotenv()
BUCKET_NAME = 'pubmed'

minio_client = Minio(
    os.environ['MINIO_API_ENDPOINT'],
    access_key=os.environ['MINIO_ACCESS_KEY'],
    secret_key=os.environ['MINIO_SECRET_KEY'],
    secure=False,
)

objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
#only the first file for testing
first_object = next(objects, None)
local_file_path = first_object.object_name
print(local_file_path)
minio_client.fget_object(BUCKET_NAME, first_object.object_name, local_file_path)
output_dir = "output_images"
pdf_to_images(local_file_path, output_dir)

# model_name = (
#     "davanstrien/finetune_colpali_v1_2-ufo-4bit"  # Use the latest version available
# )
# colpali_model = ColPali.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="cpu",  # Use "cuda:0" for GPU, "cpu" for CPU, or "mps" for Apple Silicon
# )
# colpali_processor = ColPaliProcessor.from_pretrained(
#     "vidore/colpaligemma-3b-pt-448-base"
# )
colpali_model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        # torch_dtype=torch.bfloat16,
        device_map="cpu",  # or "mps" if on Apple Silicon
    ).eval()
colpali_processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")


# sample_image_path = os.path.join(output_dir, os.listdir(output_dir)[0])
# sample_image = Image.open(sample_image_path)
# with torch.no_grad():
#     sample_batch = colpali_processor.process_images([sample_image]).to(
#         colpali_model.device
#     )
#     sample_embedding = colpali_model(**sample_batch)
# sample_embedding.shape

qdrant_client = QdrantClient(url=os.environ['QDRANT_URL'], port=os.environ['QDRANT_PORT'], https=True, api_key=os.environ['QDRANT_API_KEY'])

collection_name = "colpali"
vector_size = 128

vector_params = models.VectorParams(
    size=vector_size,
    distance=models.Distance.COSINE,
    multivector_config=models.MultiVectorConfig(
        comparator=models.MultiVectorComparator.MAX_SIM
    ),
)

scalar_quant = models.ScalarQuantizationConfig(
    type=models.ScalarType.INT8,
    quantile=0.99,
    always_ram=False,
)

qdrant_client.recreate_collection(
    collection_name=collection_name,  # the name of the collection
    on_disk_payload=True,  # store the payload on disk
    optimizers_config=models.OptimizersConfigDiff(
        indexing_threshold=100
    ),  # it can be useful to swith this off when doing a bulk upload and then manually trigger the indexing once the upload is done
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
    ),
)


@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(batch):
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        print(f"Error during upsert: {e}")
        return False
    return True

batch_size = 1  # Adjust based on your GPU memory constraints

# Use tqdm to create a progress bar
# with tqdm(total=1, desc="Indexing Progress") as pbar:
for i in range(0, 1, batch_size):
    sample_image_path = os.path.join(output_dir, os.listdir(output_dir)[0])
    images = [Image.open(sample_image_path)]
    print(type(images))
    torch.set_num_threads(torch.get_num_threads())
    print(torch.get_num_threads())
    # Process and encode images
    with torch.no_grad():
        batch_images = colpali_processor.process_images(images).to(
            colpali_model.device
        )
        image_embeddings = colpali_model(**batch_images)

    # Prepare points for Qdrant
    points = []
    for j, embedding in enumerate(image_embeddings):
        # Convert the embedding to a list of vectors
        print("multivector", embedding.shape)
        multivector = embedding.cpu().float().numpy().tolist()
        print(multivector)
        points.append(
            models.PointStruct(
                id=i + j,  # we just use the index as the ID
                vector=multivector,  # This is now a list of vectors
                payload={
                    "source": "internet archive"
                },  # can also add other metadata/data
            )
        )
    # Upload points to Qdrant
    try:
        upsert_to_qdrant(points)
        print("Batch uploaded successfully")
    # clown level error handling here 🤡
    except Exception as e:
        print(f"Error during upsert: {e}")
        continue

    # Update the progress bar
    # pbar.update(batch_size)

print("Indexing complete!")