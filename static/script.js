
/* --- MICROPHONE + WAVEFORM + WAV FILE INPUT + NEURAL BACKGROUND --- */

let startBtn = document.getElementById("startMic");
let stopBtn = document.getElementById("stopMic");

let canvas = document.getElementById("wave");
let ctx = canvas ? canvas.getContext("2d") : null;

let audioContext, analyser, microphone, dataArray, animationId;
let mediaRecorder, audioChunks = [];

/* FIX: connect recorder to hidden form input */
let fileInput = document.getElementById("hiddenAudio");

let player = document.getElementById("player");


// --- Set up waveform canvas ---
if(canvas){
    canvas.width = 600;
    canvas.height = 150;
}


// --- Start microphone ---
startBtn?.addEventListener("click", async ()=>{

    let stream = await navigator.mediaDevices.getUserMedia({audio:true});

    audioContext = new AudioContext();

    microphone = audioContext.createMediaStreamSource(stream);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;

    microphone.connect(analyser);

    let bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    drawWaveform();

    // Record audio
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

    mediaRecorder.start();
});


// --- Stop microphone ---
stopBtn?.addEventListener("click", async ()=>{

    cancelAnimationFrame(animationId);

    if(mediaRecorder && mediaRecorder.state !== "inactive"){
        mediaRecorder.stop();
    }

    mediaRecorder.onstop = async ()=>{

        const blob = new Blob(audioChunks,{type:'audio/webm'});

        const arrayBuffer = await blob.arrayBuffer();

        const buffer = await audioContext.decodeAudioData(arrayBuffer);

        const wavBlob = audioBufferToWav(buffer);

        // play audio preview
        player.src = URL.createObjectURL(wavBlob);

        // convert to file
        const file = new File([wavBlob],"recorded_audio.wav",{type:"audio/wav"});

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        // IMPORTANT: assign to hidden input
        fileInput.files = dataTransfer.files;
    };

    if(audioContext){
        audioContext.close();
    }
});


// --- Draw waveform ---
function drawWaveform(){

    animationId = requestAnimationFrame(drawWaveform);

    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "#000";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00ffff";
    ctx.beginPath();

    let slice = canvas.width / dataArray.length;
    let x = 0;

    for(let i=0;i<dataArray.length;i++){

        let v = dataArray[i] / 128.0;
        let y = v * canvas.height / 2;

        if(i===0){
            ctx.moveTo(x,y);
        }else{
            ctx.lineTo(x,y);
        }

        x += slice;
    }

    ctx.lineTo(canvas.width,canvas.height/2);
    ctx.stroke();
}


// --- Convert AudioBuffer to WAV ---
function audioBufferToWav(buffer){

    let numOfChan = buffer.numberOfChannels,
        length = buffer.length * numOfChan * 2 + 44,
        bufferArray = new ArrayBuffer(length),
        view = new DataView(bufferArray),
        channels = [],
        offset = 0,
        pos = 0,
        sample;

    function setUint16(data){
        view.setUint16(pos,data,true);
        pos += 2;
    }

    function setUint32(data){
        view.setUint32(pos,data,true);
        pos += 4;
    }

    setUint32(0x46464952);
    setUint32(length - 8);
    setUint32(0x45564157);
    setUint32(0x20746d66);
    setUint32(16);
    setUint16(1);
    setUint16(numOfChan);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numOfChan);
    setUint16(numOfChan * 2);
    setUint16(16);
    setUint32(0x61746164);
    setUint32(length - pos - 4);

    for(let i=0;i<numOfChan;i++){
        channels.push(buffer.getChannelData(i));
    }

    while(pos < length){

        for(let i=0;i<numOfChan;i++){

            sample = Math.max(-1,Math.min(1,channels[i][offset]));

            view.setInt16(
                pos,
                sample < 0 ? sample*0x8000 : sample*0x7FFF,
                true
            );

            pos += 2;
        }

        offset++;
    }

    return new Blob([bufferArray],{type:'audio/wav'});
}


// --- NEURAL NETWORK BACKGROUND ---
let net = document.getElementById("network");
let nctx = net.getContext("2d");

net.width = window.innerWidth;
net.height = window.innerHeight;

let nodes = [];

for(let i=0;i<100;i++){
    nodes.push({
        x: Math.random()*net.width,
        y: Math.random()*net.height,
        vx: (Math.random()-0.5),
        vy: (Math.random()-0.5)
    });
}

function animateNeural(){

    nctx.fillStyle="#000";
    nctx.fillRect(0,0,net.width,net.height);

    nodes.forEach(n=>{

        n.x += n.vx;
        n.y += n.vy;

        if(n.x<0 || n.x>net.width) n.vx*=-1;
        if(n.y<0 || n.y>net.height) n.vy*=-1;

        nctx.fillStyle="#00ffff";

        nctx.beginPath();
        nctx.arc(n.x,n.y,2,0,Math.PI*2);
        nctx.fill();

        nodes.forEach(m=>{

            let dist = Math.hypot(n.x-m.x,n.y-m.y);

            if(dist<120){

                nctx.strokeStyle="rgba(0,255,255,0.2)";
                nctx.beginPath();
                nctx.moveTo(n.x,n.y);
                nctx.lineTo(m.x,m.y);
                nctx.stroke();
            }

        });

    });

    requestAnimationFrame(animateNeural);
}

animateNeural();

window.addEventListener("resize",()=>{

    net.width = window.innerWidth;
    net.height = window.innerHeight;

});