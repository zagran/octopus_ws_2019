import React from "react";
import ReactDOM from "react-dom";

import * as tf from '@tensorflow/tfjs';
import "./styles.css";
import {Tensor, InferenceSession} from "onnxjs";
import {Lambda} from './interpolationLayer';

const resizeImageData = require('resize-image-data')

class App extends React.Component {
    videoRef = React.createRef();
    canvasRef = React.createRef();

    componentDidMount() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            const webCamPromise = navigator.mediaDevices
                .getUserMedia({
                    audio: false,
                    video: {
                        facingMode: "user"
                    }
                })
                .then(stream => {
                    window.stream = stream;

                    this.videoRef.current.srcObject = stream;
                    return new Promise((resolve, reject) => {
                        this.videoRef.current.onloadedmetadata = () => {
                            resolve();
                        };
                    });
                });
            this.hiddenCanvas = document.getElementById("hiddenCanvas");
            this.resultCanvas = document.getElementById("canvas");

            // link to your local address and port 3000
            const BASE_URL = 'https://192.168.0.107:3000';
            console.log(tf.getBackend());

            // let's add segmentation model path
            const segmentationModelPromise = tf.loadLayersModel(BASE_URL + '/body_model/model.json');

            // let's add styleTransfer model path
            const MODEL_URL = BASE_URL + '/style_transfer/tensorflowjs_model.pb'
            const WEIGHTS_URL = BASE_URL + '/style_transfer/weights_manifest.json'
            const styleTransferModelPromise = tf.loadGraphModel(MODEL_URL, WEIGHTS_URL)

            // now let's load models for server and run detection
            Promise.all([styleTransferModelPromise, segmentationModelPromise, webCamPromise])
                .then(values => {
                    this.laptopSegmentationalStyleTransfer(values[0], values[1]);
                })
                .then(log => console.log("Models is loaded"))
                .catch(error => {
                    console.error(error);
                });

            // link and load models
        }
    }

    laptopSegmentationalStyleTransfer = (styleTransferModel, segmentationModel) => {

        // video read

        // reading video from webcam
        var video = this.videoRef.current;
        var canvasContext = this.hiddenCanvas.getContext('2d');
        canvasContext.drawImage(video, 0, 0, video.width, video.height);
        var srcImgData = canvasContext.getImageData(0, 0, this.hiddenCanvas.width, this.hiddenCanvas.height);


        // tf.tidy converts video stream to tensor
        const image = tf.tidy(() => {
            if (!(srcImgData instanceof tf.Tensor)) {
                var frame = tf.browser.fromPixels(this.hiddenCanvas);
            }
            return frame;
        });


        // run segmentation

        // resize and normalize
        const segmentationInputSize = [256, 256]
        const segmentationInput = tf.image.resizeBilinear(image, segmentationInputSize).expandDims(0).div(255.0);

        //predict
        const segmentationMask = segmentationModel.predict(segmentationInput);
        const resizedSegmentationMask = tf.image.resizeBilinear(segmentationMask, [this.hiddenCanvas.height, this.hiddenCanvas.width]).squeeze()

        // threshold prediction
        const threshold = 0.8
        const thresholds = tf.fill([this.hiddenCanvas.height, this.hiddenCanvas.width], threshold)
        const thresholdedSegmentationMask = resizedSegmentationMask.greater(thresholds)
        const arrMask = thresholdedSegmentationMask.dataSync()


        // run style transfer
        // resize
        const slicedImage = image
        const styleTransferInputSize = [128, 170]
        const styleTransferInput = tf.image.resizeBilinear(slicedImage, styleTransferInputSize).expandDims(0).div(127.5).sub(1);

        //predict
        const styleTransferOutput = styleTransferModel.predict(styleTransferInput).add(1.).mul(127.5);

        //resize back
        const resizedStyleTransfer = tf.image.resizeBilinear(styleTransferOutput, [this.hiddenCanvas.height, this.hiddenCanvas.width])
        const resizedStyleTransferArr = resizedStyleTransfer.dataSync()


        //--

        // draw results on image

        const srcCopy = srcImgData
        // console.log(tf.memory());

        for (var i = 0; i < srcImgData.data.length; i++) {
            if (arrMask[i] !== 1) {
                srcImgData.data[i * 4] = resizedStyleTransferArr[i * 3];
                srcImgData.data[i * 4 + 1] = resizedStyleTransferArr[i * 3 + 1];
                srcImgData.data[i * 4 + 2] = resizedStyleTransferArr[i * 3 + 2];
            }

            slicedImage.dispose();
            resizedStyleTransfer.dispose();
            styleTransferOutput.dispose();
            styleTransferInput.dispose();
        }
        image.dispose();
        segmentationInput.dispose();
        segmentationMask.dispose();
        resizedSegmentationMask.dispose();
        thresholds.dispose();
        thresholdedSegmentationMask.dispose();


        var resultCanvasContext = this.resultCanvas.getContext('2d');
        resultCanvasContext.putImageData(srcImgData, 0, 0);

        requestAnimationFrame(() => {
            this.laptopSegmentationalStyleTransfer(styleTransferModel, segmentationModel);
        });
    }


    styleTransfer = () => {
        this.canvasContext.drawImage(this.video, 0, 0, this.video.width, this.video.height);
        var srcImgData = this.canvasContext.getImageData(0, 0, this.hiddenCanvas.width, this.hiddenCanvas.height);
        var input = this.imageDataToTensor(srcImgData);

        this.styleTransferSession.run([input]).then(output => {
            var styleTransferImageData = this.tensorToImageData(output, 128, 128);
            this.canvasContext.putImageData(styleTransferImageData, 0, 0);
            requestAnimationFrame(() => {
                this.styleTransfer();
            });
        })
    }

    imageDataToTensor(imageData, height, width) {
        const n = 1
        const c = 3
        const out_data = new Float32Array(n * c * height * width);
        // load src context to a tensor
        var resized = resizeImageData(imageData, width, height, 'biliniear-interpolation')
        var src_data = resized.data;

        var src_idx = 0;
        var out_idx_r = 0;
        var out_idx_g = out_idx_r + height * width;
        var out_idx_b = out_idx_g + height * width;

        const norm = 1.0;
        var src_r, src_g, src_b;
        for (var y = 0; y < height; y++) {
            for (var x = 0; x < width; x++) {
                src_r = src_data[src_idx++];
                src_g = src_data[src_idx++];
                src_b = src_data[src_idx++];
                src_idx++;

                out_data[out_idx_r++] = src_r / norm;
                out_data[out_idx_g++] = src_g / norm;
                out_data[out_idx_b++] = src_b / norm;
            }
        }

        const out = new Tensor(out_data, 'float32', [n, c, height, width]);
        return out;
    }

    tensorToImageData(tensor, height, width) {
        const outputTensor = tensor.values().next().value;
        var arr = new Uint8ClampedArray(4 * width * height);
        const h = outputTensor.dims[2];
        const w = outputTensor.dims[3];
        var t_data = outputTensor.data;

        var t_idx_r = 0;
        var t_idx_g = t_idx_r + (h * w);
        var t_idx_b = t_idx_g + (h * w);

        var dst_idx = 0;
        for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
                arr[dst_idx++] = t_data[t_idx_r++];
                arr[dst_idx++] = t_data[t_idx_g++];
                arr[dst_idx++] = t_data[t_idx_b++];
                arr[dst_idx++] = 0xFF;
            }
        }

        // Initialize a new ImageData object
        let imageData = new ImageData(arr, width, height);

        var resizedImageData = resizeImageData(imageData, this.hiddenCanvas.width, this.hiddenCanvas.height, 'biliniear-interpolation');
        return resizedImageData
    }

    render() {
        return (
            < div >
            < video
        className = "size"
        autoPlay
        playsInline
        muted
        ref = {this.videoRef}
        width = "340"
        height = "256"
        id = "video"
            / >
            < canvas
        className = "canvas size"
        ref = {this.canvasRef}
        width = "340"
        height = "256"
        id = "hiddenCanvas"
            / >
            < canvas
        className = "size"
        ref = {this.canvasRef}
        width = "340"
        height = "256"
        id = "canvas"
            / >
            < /div>
    )
        ;
    }
}

const rootElement = document.getElementById("root");
ReactDOM.render( < App / >, rootElement
)
;
