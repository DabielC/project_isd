window.onload = function() {
    Swal.fire({
        position:'top',
        title: 'This page collects image data',
        confirmButtonText: 'OK',
        showCloseButton: true,
        customClass: {
            title: 'swal-title-red',
        }
    });
};
// ฟังก์ชั่นในการเปิดกล้อง
async function openCamera() {
    Swal.fire({
        showCloseButton: true,
        title: 'Access Your Camera',
        html: `
            <video id="video" autoplay></video>
            <button id="capture-btn">Capture Photo</button>
            <canvas id="canvas" style="display:none;"></canvas>
        `,
        didOpen: async () => {
            const video = document.querySelector('#video');
            const canvas = document.querySelector('#canvas');
            const captureBtn = document.querySelector('#capture-btn');

            // Access user's camera
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            } catch (err) {
                Swal.fire('Error', 'Unable to access camera: ' + err.message, 'error');
                return;
            }

            // ฟังก์ชั่นถ่ายรูป
            captureBtn.addEventListener('click', () => {
                const context = canvas.getContext('2d');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // flipรูป
                context.translate(canvas.width, 0);
                context.scale(-1, 1);
                context.drawImage(video, 0, 0);
                const imageUrl = canvas.toDataURL('image/png');

                // แสดงรูปที่ถ่าย
                Swal.fire({
                    title: 'Captured Photo',
                    imageUrl: imageUrl,
                    imageAlt: 'Captured Photo',
                    showDenyButton: true,
                    confirmButtonText: 'OK',
                    denyButtonText: 'Retake',
                    showCloseButton: true,
                }).then((result) => {
                    // ถ้า user กด OK
                    if (result.isConfirmed) {
                        // ส่งรูปไป ฟังก์ชั่น sendImageToBackend
                        sendImageToBackend(imageUrl);
                    } else if (result.isDenied) {
                        // ถ่ายใหม่
                        openCamera();
                    }
                });
            });
        },
        willClose: () => {
            // ปิดกล้อง
            const video = document.querySelector('#video');
            const stream = video.srcObject;
            if (stream) {
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
            }
            video.srcObject = null;
        },
        showConfirmButton: false,
        width: '80vw',
    });
}

// ทำเป็น base64 และส่งไป backend
function fileToImage(event) {
    const input = event.target;
    const file = input.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            // ทำเป็น base64
            const imageUrl = e.target.result;

            // แสดงรูป
            Swal.fire({
                title: 'Uploaded Image',
                imageUrl: imageUrl,
                imageAlt: 'Uploaded Image',
                confirmButtonText: 'OK',
                showCloseButton: true,
            }).then((result) => {
                if (result.isConfirmed) {
                    // ส่งรูปไป sendImageToBackend
                    sendImageToBackend(imageUrl);
                }
            });
            
        };
        reader.readAsDataURL(file);
    }
}

// ฟังก์ชั่นในการส่งข้อมูลไป API โดยใช้ POST 
function sendImageToBackend(imageUrl) {
    const data = {
        image: imageUrl
    };

    Swal.fire({
        title: 'Loading...',
        text: 'Please wait while we process your request.',
        allowOutsideClick: false,
        didOpen: () => {
            Swal.showLoading();
        }
    });

    fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                // ข้อมูลที่ไม่ใช่รูปจะแสดงผล pop up โดยมีข้อความว่า Please input the image
                Swal.fire({
                    icon: 'error',
                    title: 'Error',
                    text: 'Please input the image',
                });
                throw new Error(err.detail);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log(`Success:`, data);
        // หากไม่พบหน้า จะแสดง pop up ว่า No face detected
        
        if (data === "No face detected") {
            Swal.fire({
                icon: 'error',
                title: 'Oops...',
                text: 'No face detected',
            });
        } 
        // ส่งข้อมูลไปหน้า test.html
        else {
            const params = new URLSearchParams({
                data: JSON.stringify(data)
            });
            window.location.href = `test.html?${params.toString()}`;
        }
    })
    .catch((error) => {
        console.error(`Error from:`, error);
    });
     
}