const classes = ['Japanese Geta', 'Litchi', 'Pens', 'Squirrel', 'Tote Bag'];
const resultEl = document.getElementById("result");
const previewEl = document.getElementById("preview");

let session;
async function initONNX() {
    session = await ort.InferenceSession.create("cnn_model.onnx");
    console.log("ONNX model loaded!");
}
initONNX();

function preprocess(img) {
    // Resize and normalize the image
    const canvas = document.createElement("canvas");
    canvas.width = 128; canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 128, 128);
    const imgData = ctx.getImageData(0, 0, 128, 128);
    const data = new Float32Array(3*128*128);
    for(let i=0;i<128*128;i++){
        data[i]      = (imgData.data[i*4]/255 - 0.5)/0.5; // R
        data[i+128*128] = (imgData.data[i*4+1]/255 - 0.5)/0.5; // G
        data[i+2*128*128] = (imgData.data[i*4+2]/255 - 0.5)/0.5; // B
    }
    return new ort.Tensor("float32", data, [1,3,128,128]);
}

async function predict() {
    const inputFile = document.getElementById("imgInput").files[0];
    if(!inputFile){ alert("Select an image!"); return; }

    const img = new Image();
    img.onload = async () => {
        previewEl.src = img.src;
        const inputTensor = preprocess(img);
        const feeds = { input: inputTensor };
        const output = await session.run(feeds);
        const pred = output.output.data;
        const idx = pred.indexOf(Math.max(...pred));
        resultEl.textContent = "Predicted class: " + classes[idx];
    };
    img.src = URL.createObjectURL(inputFile);
}
