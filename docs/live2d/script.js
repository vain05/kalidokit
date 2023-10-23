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
const modelUrl = "../models/DemoKit5/DemoKit5.model3.json";

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

        // // eye blink
        // coreModel.setParameterValueById("Param14", stabilizedEyes.l);
        // coreModel.setParameterValueById("ParamEyeROpen", stabilizedEyes.r);

        // console.log(blendshapes)


//         // brownInnerUp
//         coreModel.setParameterValueById(
//             "Param",
//             blendshapes[3].score
//         )

//         // // brownDownLeft
//         // coreModel.setParameterValueById(
//         //     "Param2",
//         //     blendshapes[1].score * 35
//         // )


//         // // brownDownRight
//         // coreModel.setParameterValueById(
//         //     "Param3",
//         //     blendshapes[2].score * 35
//         // )

//         if (blendshapes[1].score > blendshapes[2].score)
//             coreModel.setParameterValueById(
//                 "Param2",
//                 blendshapes[1].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param2",
//                 - blendshapes[2].score
//             )
        
//         // // brownOutterUpLeft
//         // coreModel.setParameterValueById(
//         //     "Param4",
//         //     blendshapes[4].score * 35
//         // )

//         // // brownOutterUpRight
//         // coreModel.setParameterValueById(
//         //     "Param5",
//         //     blendshapes[5].score * 35
//         // )

//         if (blendshapes[4].score > blendshapes[5].score)
//             coreModel.setParameterValueById(
//                 "Param3",
//                 blendshapes[4].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param3",
//                 - blendshapes[5].score
//             )

//         // // eyeLookUpLeft
//         // coreModel.setParameterValueById(
//         //     "Param6",
//         //     blendshapes[17].score * 35
//         // )

//         // // eyeLookUpRight
//         // coreModel.setParameterValueById(
//         //     "Param7",
//         //     blendshapes[18].score * 35
//         // )

//         // // eyeLookDownLeft
//         // coreModel.setParameterValueById(
//         //     "Param8",
//         //     blendshapes[11].score * 35
//         // )

//         // // eyeLookDownRight
//         // coreModel.setParameterValueById(
//         //     "Param9",
//         //     blendshapes[12].score * 35
//         // )
        
//         if (blendshapes[17].score > blendshapes[11].score)
//             coreModel.setParameterValueById(
//                 "Param4",
//                 blendshapes[17].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param4",
//                 - blendshapes[11].score
//             )        


//         if (blendshapes[18].score > blendshapes[12].score)
//             coreModel.setParameterValueById(
//                 "Param4",
//                 blendshapes[18].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param4",
//                 - blendshapes[12].score
//             )
        
//         // // eyeLookInLeft
//         // coreModel.setParameterValueById(
//         //     "Param10",
//         //     blendshapes[13].score * 35
//         // )

//         // // eyeLookInRight
//         // coreModel.setParameterValueById(
//         //     "Param11",
//         //     blendshapes[14].score * 35
//         // )

//         // // eyeLookOutLeft
//         // coreModel.setParameterValueById(
//         //     "Param12",
//         //     blendshapes[15].score * 35
//         // )

//         // // eyeLookOutRight
//         // coreModel.setParameterValueById(
//         //     "Param13",
//         //     blendshapes[16].score * 35
//         // )

//         if (blendshapes[15].score > blendshapes[16].score)
//             coreModel.setParameterValueById(
//                 "Param5",
//                 blendshapes[15].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param5",
//                 - blendshapes[16].score
//             )
 
//         // eyeBlinkLeft
//         coreModel.setParameterValueById(
//             "Param7",
//             blendshapes[9].score * 1.5
//         )

//         // eyeBlinkRight
//         coreModel.setParameterValueById(
//             "Param6",
//             blendshapes[10].score * 1.5
//         )

//         // // eyeSquintLeft
//         // coreModel.setParameterValueById(
//         //     "Param16",
//         //     blendshapes[19].score * 35
//         // )

//         // // eyeSquintRight
//         // coreModel.setParameterValueById(
//         //     "Param17",
//         //     blendshapes[20].score * 35
//         // )

//         // jawOpen
//         coreModel.setParameterValueById(
//             "Param8",
//             blendshapes[25].score
//         )

//         // // jawLeft
//         // coreModel.setParameterValueById(
//         //     "Param27",
//         //     blendshapes[24].score * 35
//         // )

//         // // jawRight
//         // coreModel.setParameterValueById(
//         //     "Param28",
//         //     blendshapes[26].score * 35
//         // )

//         if (blendshapes[24].score > blendshapes[26].score)
//             coreModel.setParameterValueById(
//                 "Param9",
//                 blendshapes[24].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param9",
//                 - blendshapes[26].score
//             )

//         // // mouthFunnel
//         // coreModel.setParameterValueById(
//         //     "Param29",
//         //     blendshapes[32].score * 35
//         // )

//         // // mouthPucker
//         // coreModel.setParameterValueById(
//         //     "Param30",
//         //     blendshapes[38].score * 35
//         // )

//         // // mouthLeft
//         // coreModel.setParameterValueById(
//         //     "Param31",
//         //     blendshapes[33].score * 35
//         // )

//         // // mouthRight
//         // coreModel.setParameterValueById(
//         //     "Param32",
//         //     blendshapes[39].score * 35
//         // )

//         if (blendshapes[33].score > blendshapes[39].score)
//             coreModel.setParameterValueById(
//                 "Param10",
//                 blendshapes[33].score
//             )
//         else 
//             coreModel.setParameterValueById(
//                 "Param10",
//                 - blendshapes[39].score
//             )

//         // mouthSmileLeft
//         coreModel.setParameterValueById(
//             "Param11",
//             blendshapes[44].score
//         )
           
//         // mouthSmileRight
//         coreModel.setParameterValueById(
//             "Param12",
//             blendshapes[45].score
//         )
              
//         // // mouthDimmpleLeft
//         // coreModel.setParameterValueById(
//         //     "Param42",
//         //     blendshapes[28].score * 35
//         // )
       
//         // // mouthDimmpleRight
//         // coreModel.setParameterValueById(
//         //     "Param43",
//         //     blendshapes[29].score * 35
//         // )

//         // mouthUpperUpLeft
//         coreModel.setParameterValueById(
//             "Param13",
//             blendshapes[48].score
//         )
       
//         // mouthUpperUpRight
//         coreModel.setParameterValueById(
//             "Param14",
//             blendshapes[49].score
//         )

//         // mouthLowerDownLeft
//         coreModel.setParameterValueById(
//             "Param15",
//             blendshapes[34].score
//         )
       
//         // mouthLowerDownRight
//         coreModel.setParameterValueById(
//             "Param16",
//             blendshapes[35].score
//         )
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
