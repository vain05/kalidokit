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
let modelUrl = "../models/DemoKit4/DemoKit4.model3.json";

let currentModel, faceLandmarker;

function addDirectory(item) {
    if (item.isDirectory) {
        var directoryReader = item.createReader();
        directoryReader.readEntries(function(entries) {
        entries.forEach(function(entry) {
                addDirectory(entry);
            });
        });
    } else {
        item.file(function(file){
            if (file.name.includes("model3")) {
                modelUrl = file.name
                let file_path = URL.createObjectURL(file)
                console.log(file_path)
            }
        });
    }
}

function dropHandler(ev) {
    console.log("File(s) dropped");

    // Prevent default behavior (Prevent file from being opened)
    ev.preventDefault();

    if (ev.dataTransfer.items) {
        // Use DataTransferItemList interface to access the file(s)
        [...ev.dataTransfer.items].forEach((item, i) => {
            const entry = item.webkitGetAsEntry();
            console.log(`… file[${i}].name = ${entry.name}`);

            if (entry) {
                addDirectory(entry);
            }

        });
    } else {
        // Use DataTransfer interface to access the file(s)
        [...ev.dataTransfer.files].forEach((file, i) => {
            console.log(`… file[${i}].name = ${file.name}`);
        });
    }
}
  

window.dropHandler = dropHandler;

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
            "Angl3X",
            lerp(result.head.degrees.y / 10, coreModel.getParameterValueById("Angl3X"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "Angl3Y",
            lerp(result.head.degrees.x / 10, coreModel.getParameterValueById("Angl3Y"), lerpAmount)
        );
        coreModel.setParameterValueById(
            "Angl3Z",
            lerp(result.head.degrees.z, coreModel.getParameterValueById("Angl3Z"), lerpAmount)
        );

        // update body params for models without head/body param sync
        // const dampener = 0.3;
        // coreModel.setParameterValueById(
        //     "ParamBodyAngleX",
        //     lerp(result.head.degrees.y * dampener * 5, coreModel.getParameterValueById("ParamBodyAngleX"), lerpAmount)
        // );
        // coreModel.setParameterValueById(
        //     "ParamBodyAngleY",
        //     lerp(result.head.degrees.x * dampener * 5, coreModel.getParameterValueById("ParamBodyAngleY"), lerpAmount)
        // );
        // coreModel.setParameterValueById(
        //     "ParamBodyAngleZ",
        //     lerp(result.head.degrees.z * dampener * 5, coreModel.getParameterValueById("ParamBodyAngleZ"), lerpAmount)
        // );

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

        coreModel.setParameterValueById(
            "bronwInnerUp",
            blendshapes[3].score
        )

        if (blendshapes[1].score > blendshapes[2].score)
            coreModel.setParameterValueById(
                "brownDownLeft_Right",
                blendshapes[1].score
            )
        else 
            coreModel.setParameterValueById(
                "brownDownLeft_Right",
                - blendshapes[2].score
            )
        
        if (blendshapes[4].score > blendshapes[5].score)
            coreModel.setParameterValueById(
                "browOuterUpLeft_Right",
                blendshapes[4].score
            )
        else 
            coreModel.setParameterValueById(
                "browOuterUpLeft_Right",
                - blendshapes[5].score
            )
      
        if (blendshapes[17].score > blendshapes[11].score)
            coreModel.setParameterValueById(
                "eyeLookUp_Down",
                lerp(blendshapes[17].score, 
                     coreModel.getParameterValueById("eyeLookUp_Down"), 
                     lerpAmount)
            )
        else 
            coreModel.setParameterValueById(
                "eyeLookUp_Down",
                - lerp(blendshapes[11].score, 
                       - coreModel.getParameterValueById("eyeLookUp_Down"), 
                       lerpAmount)
            )        


        if (blendshapes[18].score > blendshapes[12].score)
            coreModel.setParameterValueById(
                "eyeLookUp_Down",
                lerp(blendshapes[18].score, 
                     coreModel.getParameterValueById("eyeLookUp_Down"), 
                     lerpAmount)
            )
        else 
            coreModel.setParameterValueById(
                "eyeLookUp_Down",
                - lerp(blendshapes[12].score, 
                       - coreModel.getParameterValueById("eyeLookUp_Down"), 
                       lerpAmount)
            )
        
        if (blendshapes[15].score > blendshapes[16].score)
            coreModel.setParameterValueById(
                "eyeLookOutLeft_Right",
                lerp(blendshapes[15].score, 
                     coreModel.getParameterValueById("eyeLookOutLeft_Right"), 
                     lerpAmount)
                
            )
        else 
            coreModel.setParameterValueById(
                "eyeLookOutLeft_Right",
                - lerp(blendshapes[16].score, 
                       - coreModel.getParameterValueById("eyeLookOutLeft_Right"), 
                       lerpAmount)
            )
 
        // eyeBlinkLeft
        coreModel.setParameterValueById(
            "eyeBlinkLeft",       
            lerp(blendshapes[9].score * 1.5, 
                 coreModel.getParameterValueById("eyeBlinkLeft"), 
                 lerpAmount)
        )

        // eyeBlinkRight
        coreModel.setParameterValueById(
            "eyeBlinkRight",
            blendshapes[10].score * 1.5,
            lerp(blendshapes[10].score * 1.5, 
                 coreModel.getParameterValueById("eyeBlinkRight"), 
                 lerpAmount)
        )

        // jawOpen
        coreModel.setParameterValueById(
            "jawOpen",
            blendshapes[25].score
        )

        if (blendshapes[24].score > blendshapes[26].score)
            coreModel.setParameterValueById(
                "jawLeft_Right",
                blendshapes[24].score
            )
        else 
            coreModel.setParameterValueById(
                "jawLeft_Right",
                - blendshapes[26].score
            )

        if (blendshapes[33].score > blendshapes[39].score)
            coreModel.setParameterValueById(
                "mouthLeft_Right",
                blendshapes[33].score
            )
        else 
            coreModel.setParameterValueById(
                "mouthLeft_Right",
                - blendshapes[39].score
            )

        // mouthSmileLeft
        coreModel.setParameterValueById(
            "mouthSmileLeft",
            blendshapes[44].score
        )
           
        // mouthSmileRight
        coreModel.setParameterValueById(
            "mouthSmileRight",
            blendshapes[45].score
        )
       
        // mouthUpperUpLeft
        coreModel.setParameterValueById(
            "mouthUpperUpLeft",
            blendshapes[48].score
        )
       
        // mouthUpperUpRight
        coreModel.setParameterValueById(
            "mouthUpperUpRight",
            blendshapes[49].score
        )

        // mouthLowerDownLeft
        coreModel.setParameterValueById(
            "mouthLowerDownLeft",
            blendshapes[34].score
        )
       
        // mouthLowerDownRight
        coreModel.setParameterValueById(
            "mouthLowerDownRight",
            blendshapes[35].score
        )
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
