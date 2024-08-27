from flask_cors import CORS, cross_origin
from flask import Flask, send_file
from PIL import Image
import flask
import numpy as np
import cv2
import os
import onnxruntime



import onnxruntime
import os

# Replace 'your_absolute_path' with the actual absolute path to the .onnx file
model_path = "D:/Visual app/Visual-main/Backend1/sam_vit_b_encoder.onnx"
if os.path.exists(model_path):
    encoder_session = onnxruntime.InferenceSession(model_path)
else:
    print(f"Model file not found at {model_path}")



# encoder_path = "sam_vit_b_encoder.onnx"
# encoder_session = onnxruntime.InferenceSession(
#     encoder_path, providers=['CPUExecutionProvider'])

app = Flask(__name__)
CORS(app, expose_headers=["Content-Disposition"])


@app.route('/', methods=['GET'])
@cross_origin(origin='*')
def home():
    return "Hello, World !!"

@app.route('/test',methods=['POST'])
def testpost():
    imagfile=flask.request.files['image']
    print(imagfile)
    return "Image Recieved"

@app.route('/getembedding', methods=['POST'])
@cross_origin(origin='*')
def getembedding1():
    imagefile = flask.request.files['image']
    print(imagefile)
    img = Image.open(imagefile)
    cv_image = np.array(img)
    input_size = (684, 1024)
    scale_x = input_size[1] / cv_image.shape[1]
    scale_y = input_size[0] / cv_image.shape[0]
    scale = min(scale_x, scale_y)
    transform_matrix = np.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1],
        ]
    )
    cv_image = cv2.warpAffine(
        cv_image,
        transform_matrix[:2],
        (input_size[1], input_size[0]),
        flags=cv2.INTER_LINEAR,
    )
    encoder_inputs = {
        "input_image": cv_image.astype(np.float32),
    }
    output = encoder_session.run(None, encoder_inputs)
    if output is not None:
        output = output[0]
    image_embedding = output
    return flask.jsonify(image_embedding.tolist())


if (__name__ == '__main__'):
    app.run(debug=True,port=os.getenv("PORT", default=10000),host=os.getenv("HOST", default="0.0.0.0"))












# from flask_cors import CORS
# from flask import Flask, send_file, request
# from PIL import Image
# import numpy as np
# import os
# from segment_anything import sam_model_registry, SamPredictor

# checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device='cpu')
# predictor = SamPredictor(sam)

# app = Flask(__name__)
# CORS(app)

# @app.route('/', methods=['GET'])
# def home():
#     return "Hello, World !!"

# @app.route('/getembedding', methods=['POST'])
# def getembedding():
#     imagefile = request.files['image']
#     img = Image.open(imagefile)
#     img = np.array(img, dtype=np.float32)
#     predictor.set_image(img)
#     image_embedding = predictor.get_image_embedding().cpu().numpy()

#     # Desired dimensions
#     desired_shape = (1, 256, 64, 64)
#     desired_size = np.prod(desired_shape)  # 1 * 256 * 64 * 64 = 1048576

#     # Resize or pad the embedding to the desired size
#     original_size = image_embedding.size
#     if original_size > desired_size:
#         # Truncate the embedding if it is larger than the desired size
#         image_embedding = image_embedding.flat[:desired_size]
#     elif original_size < desired_size:
#         # Pad the embedding if it is smaller than the desired size
#         padded_embedding = np.zeros(desired_size, dtype=np.float32)
#         padded_embedding[:original_size] = image_embedding.flat
#         image_embedding = padded_embedding

#     # Ensure the Embeddings directory exists
#     directory = "Embeddings"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
    
#     # Save the embedding
#     a = np.random.randint(100)
#     filepath = os.path.join(directory, f"image_embedding{a}.npy")
#     np.save(filepath, image_embedding)
    
#     print(filepath, image_embedding)
#     return send_file(filepath, mimetype='application/octet-stream', as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=False, port=os.getenv("PORT", default=5000), host=os.getenv("HOST", default="0.0.0.0"))
