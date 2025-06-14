import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import kNear from "./knear.js";

const enableWebcamButton = document.getElementById("webcamButton")

const video = document.getElementById("webcam")
const canvasElement = document.getElementById("output_canvas")
const canvasCtx = canvasElement.getContext("2d")

const drawUtils = new DrawingUtils(canvasCtx)
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;
let classifier =  new kNear(3);

fetch('./model.json')
    .then(res => res.json())
    .then(modelData => {
        // Loop through and feed into classifier
        modelData.forEach(entry => {
            classifier.learn(entry.data, entry.label);
        });
        console.log("Model loaded into classifier");
    })
    .catch(err => {
        console.error("Failed to load model.json:", err);
    });

const tenchijinImages = {
    "Ten": "images/ten.jpg",
    "Chi": "images/chi.jpg",
    "Jin": "images/jin.jpg"
}

let image = document.querySelector("#myimage")
// const collectedData = [];

/********************************************************************
 // CREATE THE POSE DETECTOR
 ********************************************************************/
const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });
    console.log("model loaded, you can start webcam")

    enableWebcamButton.addEventListener("click", (e) => enableCam(e))
}

/********************************************************************
 // START THE WEBCAM
 ********************************************************************/

// Start webcam and draw a canvas over de video.
async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

/********************************************************************
 // START PREDICTIONS
 ********************************************************************/

// Wait for app to detect camera for video.
// if a hand is visible mark the vectors (locations of fingers/bones structure).
// start tracking from the thumb.
async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now())

    let hand = results.landmarks[0]
    if(hand) {
        if (classifier.training.length > 0){
            const inputVector = hand.flatMap(p => [p.x, p.y, p.z])
            const prediction = classifier.classify(inputVector)

            image.innerText = prediction; // Keep if this is also useful for something

            const display = document.getElementById("predictionDisplay");
            display.innerHTML = ""; // Clear any previous content

            if (tenchijinImages[prediction]) {
                const img = document.createElement("img");
                img.src = tenchijinImages[prediction];
                img.alt = prediction;
                img.style.maxWidth = "100%";
                img.style.maxHeight = "100%";
                display.appendChild(img); // Show Image when matching hand sign.

                const span = document.createElement("span");
                span.innerText = `${prediction}`;
                span.style.color = "white";
                display.appendChild(span); // Show Text when matching hand sign.
            } else {
                display.innerText = "Unknown Sign"; // Show text when hand sign is not registered.
            }

            const overlay = document.getElementById("overlayImage");
            if (tenchijinImages[prediction]) {
                overlay.style.backgroundImage = `url('${tenchijinImages[prediction]}')`;
            } else {
                // overlay.style.backgroundImage = ""; // clear if unknown
            }
        } else {
            // No hand detected — clear both displays
            // const overlay = document.getElementById("overlayImage");
            // overlay.style.backgroundImage = "";
            //
            // const display = document.getElementById("predictionDisplay");
            // display.innerHTML = "<span style='color: white;'>No hand detected</span>";
        }

        let thumb = hand[4]
        image.style.transform = `translate(${video.videoWidth - thumb.x * video.videoWidth}px, ${thumb.y * video.videoHeight}px)`
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    for(let hand of results.landmarks){
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam)
    }
}

/********************************************************************
 // START THE APP
 ********************************************************************/
if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker()
}
