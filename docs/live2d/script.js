import Vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const { FaceLandmarker, FilesetResolver, DrawingUtils } = Vision;

const {
    Application,
    live2d: { Live2DModel },
} = PIXI;

// Kalidokit provides a simple easing function
// (linear interpolation) used for animation smoothness
// you can use a more advanced easing function if you want
const {
    Face,
    Vector: { lerp },
    Utils: { clamp },
} = Kalidokit;

const videoElement = document.querySelector(".input_video"),
    guideCanvas = document.querySelector("canvas.guides");

// Url to Live2D
const modelUrl = "../models/Mao/Mao.model3.json";

let currentModel, faceLandmarker;

(async function main() {
    // create pixi application
    const app = new PIXI.Application({
        view: document.getElementById("live2d"),
        autoStart: true,
        backgroundAlpha: 0,
        backgroundColor: 0xffffff,
        resizeTo: window,
    });

    // load live2d model
    currentModel = await Live2DModel.from(modelUrl, { autoInteract: false });
    currentModel.scale.set(0.4);
    currentModel.interactive = true;
    currentModel.anchor.set(0.5, 0.5);
    currentModel.position.set(window.innerWidth * 0.5, window.innerHeight * 0.8);

    // Add events to drag model
    currentModel.on("pointerdown", (e) => {
        currentModel.offsetX = e.data.global.x - currentModel.position.x;
        currentModel.offsetY = e.data.global.y - currentModel.position.y;
        currentModel.dragging = true;
    });
    currentModel.on("pointerup", (e) => {
        currentModel.dragging = false;
    });
    currentModel.on("pointermove", (e) => {
        if (currentModel.dragging) {
            currentModel.position.set(e.data.global.x - currentModel.offsetX, e.data.global.y - currentModel.offsetY);
        }
    });

    // Add mousewheel events to scale model
    document.querySelector("#live2d").addEventListener("wheel", (e) => {
        e.preventDefault();
        currentModel.scale.set(clamp(currentModel.scale.x + event.deltaY * -0.001, -0.5, 10));
    });

    // add live2d model to stage
    app.stage.addChild(currentModel);

    const filesetResolver = await FilesetResolver.forVisionTasks(
        // path/to/wasm/root
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(
        filesetResolver,
        {
            baseOptions: {
                modelAssetPath: "app/shared/models/face_landmarker.task",
                delegate: "GPU",
            },
            runningMode: "VIDEO",
            outputFaceBlendshapes: true,
            numFaces: 1,
        }
    );

    startCamera();
})();

const onResults = (results) => {
    animateLive2DModel(results.faceLandmarks[0], results.faceBlendshapes[0]);
};

const animateLive2DModel = (landmarks, blendshapes) => {
    if (!currentModel || !landmarks) return;

    let riggedFace;

    // use kalidokit face solver
    riggedFace = Face.solve(landmarks, {
        runtime: "mediapipe",
        video: videoElement,
    });

    rigFace(riggedFace, blendshapes.categories, 0.5);
};

// update live2d model internal state
const rigFace = (result, blendshapes, lerpAmount = 0.7) => {
    if (!currentModel || !result) return;
    const coreModel = currentModel.internalModel.coreModel;

    currentModel.internalModel.motionManager.update = (...args) => {
        // disable default blink animation
        currentModel.internalModel.eyeBlink = undefined;

        coreModel.setParameterValueById(
            "ParamEyeBallX",
            lerp(result.pupil.x, coreModel.getParameterValueById("ParamEyeBallX"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "ParamEyeBallY",
            lerp(result.pupil.y, coreModel.getParameterValueById("ParamEyeBallY"), lerpAmount)
        );

        // X and Y axis rotations are swapped for Live2D parameters
        // because it is a 2D system and KalidoKit is a 3D system
        coreModel.setParameterValueById(
            "ParamAngleX",
            lerp(result.head.degrees.y, coreModel.getParameterValueById("ParamAngleX"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "ParamAngleY",
            lerp(result.head.degrees.x, coreModel.getParameterValueById("ParamAngleY"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "ParamAngleZ",
            lerp(result.head.degrees.z, coreModel.getParameterValueById("ParamAngleZ"), lerpAmount)
        );

        // update body params for models without head/body param sync
        const dampener = 0.3;
        coreModel.setParameterValueById(
            "ParamBodyAngleX",
            lerp(result.head.degrees.y * dampener, coreModel.getParameterValueById("ParamBodyAngleX"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "ParamBodyAngleY",
            lerp(result.head.degrees.x * dampener, coreModel.getParameterValueById("ParamBodyAngleY"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "ParamBodyAngleZ",
            lerp(result.head.degrees.z * dampener, coreModel.getParameterValueById("ParamBodyAngleZ"), lerpAmount)
        );

        // Simple example without winking.
        // Interpolate based on old blendshape, then stabilize blink with `Kalidokit` helper function.
        let stabilizedEyes = Kalidokit.Face.stabilizeBlink(
            {
                l: lerp(result.eye.l, coreModel.getParameterValueById("ParamEyeLOpen"), 0.7),
                r: lerp(result.eye.r, coreModel.getParameterValueById("ParamEyeROpen"), 0.7),
            },
            result.head.y
        );
        // eye blink
        coreModel.setParameterValueById("ParamEyeLOpen", stabilizedEyes.l);
        coreModel.setParameterValueById("ParamEyeROpen", stabilizedEyes.r);

        // // mouth
        // coreModel.setParameterValueById(
        //     "ParamMouthOpenY",
        //     lerp(result.mouth.y, coreModel.getParameterValueById("ParamMouthOpenY"), 0.3)
        // );
        // // Adding 0.3 to ParamMouthForm to make default more of a "smile"
        // coreModel.setParameterValueById(
        //     "ParamMouthForm",
        //     0.3 + lerp(result.mouth.x, coreModel.getParameterValueById("ParamMouthForm"), 0.3)
        // );

        // mouth

        // jawOpen
        coreModel.setParameterValueById(
            "ParamMouthA",
            // blendshapes[25].score
            1
        );

        // // mouthSmileLeft and mounthSmileRight
        // const mouth_i = (blendshapes[44].score + blendshapes[45].score + blendshapes[48].score + blendshapes[49].score) / 4;
        // coreModel.setParameterValueById(
        //     "ParamMouthI",
        //     mouth_i
        // );

        // // mouthPucker
        // coreModel.setParameterValueById(
        //     "ParamMouthU",
        //     blendshapes[38].score
        // );

        // // console.log(coreModel.getParameterValueById("ParamMouthU"))

        // coreModel.setParameterValueById(
        //     "ParamMouthE",
        //     (blendshapes[44].score + blendshapes[45].score) / 2
        // );

        // coreModel.setParameterValueById(
        //     "ParamMouthO",
        //     0.5
        // );
        // console.log()
    };
};

// start camera using mediapipe camera utils
const startCamera = () => {
    const camera = new Camera(videoElement, {
        onFrame: () => {
            let nowInMs = Date.now();
            const results = faceLandmarker.detectForVideo(videoElement, nowInMs);
            onResults(results);
        },
        width: 640,
        height: 480,
    });

    camera.start();
};
