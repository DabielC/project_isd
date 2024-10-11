window.onload = function() {
    const urlParams = new URLSearchParams(window.location.search);
    // รับข้อมูลที่ส่งผ่าน URL
    const dataString = urlParams.get('data');
    //สร้างตัวแปร like dislike button
    let likeButton = document.getElementById('likeButton');
    let dislikeButton = document.getElementById('dislikeButton');

    //เมื่อคลิกแล้วปุ่มจะเปลี่ยนสีและเข้าฟังก์ชั่น 
    likeButton.addEventListener('click', function(event) {
        event.preventDefault(); 
        likeButton.classList.add('bg-gray-600');
        dislikeButton.classList.remove('bg-gray-600');
        if (dataString) {
            const parsedData = JSON.parse(dataString);
            updateReflectionLike(parsedData.id, 1); // 1 สำหรับ Like
        }
    });

    dislikeButton.addEventListener('click', function(event) {
        event.preventDefault(); 
        dislikeButton.classList.add('bg-gray-600');
        likeButton.classList.remove('bg-gray-600');
        if (dataString) {
            const parsedData = JSON.parse(dataString);
            updateReflectionLike(parsedData.id, 0); // 0 สำหรับ Dislike
        }
    });
    const imageElement = document.getElementById('image');

    imageElement.addEventListener('click', toggleVisibility);

    //ถ้ามีข้อมูลจะทำงานดังต่อไปนี้
    if (dataString) {
        try {
            const parsedData = JSON.parse(dataString); // แปลงกลับเป็น object
            console.log('Parsed Data:', parsedData);

            // แสดงผล face shape จาก vscore
            if (parsedData.vscore && parsedData.vscore.class) {
                const faceShapeElement = document.getElementById('face-shape-result');
                faceShapeElement.textContent = `${parsedData.vscore.class}`;
            }

            // แสดงกราฟ
            if (parsedData.vscore && parsedData.vscore.score) {
                displayResults(parsedData.vscore.score);
            }

            // แสดงผล glasses ที่แนะนำ
            const mostSuitableGlasses = parsedData.products.filter(product => product.suitability === "Most Suitable");
            const appropriateGlasses = parsedData.products.filter(product => product.suitability === "Appropriate");
            const highlyAppropriateGlasses = parsedData.products.filter(product => product.suitability === "Highly Appropriate");
            
            // แสดงผลใน div ที่ต้องการ
            const mostSuitableContainer = document.getElementById("most-suitable-container");
            const suitableContainer = document.getElementById("suitable");
            
            // ลบค่าซ้ำด้วย Set แล้วแปลงกลับเป็นอาร์เรย์และรวมผลลัพธ์เป็นสตริง
            mostSuitableContainer.innerHTML = [...new Set(appropriateGlasses.map(glass => glass.glasses_type))].join(', ');
            suitableContainer.innerHTML = [...new Set(highlyAppropriateGlasses.map(glass => glass.glasses_type))].join(', ');                    

            // แสดงแว่นตาแนะนำ
            if (parsedData.products) {
                displayGlassesRecommendations(parsedData.products);
            }

            const itemId = parsedData.id; // ดึง ID จาก parsedData
            fetchImage(itemId); // เรียกฟังก์ชันเพื่อดึงภาพ

        } catch (error) {
            console.error('Error parsing data:', error);
        }
    }
    
    //รับข้อมูลภาพที่user input มาผ่าน get
    function fetchImage(itemId) {
        fetch(`http://127.0.0.1:8000/get_image/${itemId}`)
            .then(response => response.json())
            .then(imageData => {
                console.log(imageData)
                if (imageData) {
                    const imageElement = document.getElementById('image'); // สมมติว่ามี <img> ที่จะใช้แสดงภาพ
                    imageElement.src = `${imageData}`;
                    
                } else {
                    console.error('Image not found');
                }
            })
            .catch(error => {
                console.error('Error fetching image:', error);
            });
    }

    let isChartVisible = true;

    //เมื่อคลิกกราฟแล้วจะเปลี่ยนเป็นรูปภาพ
    function toggleVisibility() {
        const chartElement = document.getElementById('vote-chart');
        const imageElement = document.getElementById('image');

        if (isChartVisible) {
            chartElement.style.display = 'none'; // Hide the chart
            imageElement.style.display = 'block'; // Show the image
        } else {
            chartElement.style.display = 'block'; // Show the chart
            imageElement.style.display = 'none'; // Hide the image
        }

        isChartVisible = !isChartVisible;
    }

    document.getElementById('vote-chart').addEventListener('click', toggleVisibility);

    imageElement.addEventListener('click', toggleVisibility);

    document.getElementById('commentForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const commentText = document.getElementById('comment').value; // รับข้อความคอมเมนต์
        if (dataString) {
            const parsedData = JSON.parse(dataString);
            updateReflectionComment(parsedData.id, commentText); // อัพเดตคอมเมนต์
        }
    });
}

// ฟังก์ชันสำหรับอัพเดตสถานะ Like/Dislike แล้วส่งไป API
function updateReflectionLike(id, likeStatus) {
    event.preventDefault();
    fetch('http://127.0.0.1:8000/update_reflection_like', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            id: id,
            like: likeStatus
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Update successful:', data);
        Swal.fire({
            icon: 'success',
            title: 'Update Successful!',
            confirmButtonText: 'OK'
        });
    })
    .catch(error => {
        console.error('Error updating like/dislike:', error);
    });
}

// ฟังก์ชันสำหรับอัพเดต comment แล้วส่งไป API
function updateReflectionComment(id, comment) {
    event.preventDefault();
    fetch('http://127.0.0.1:8000/update_reflection_comment', {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            id: id,
            comment: comment
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Comment updated successfully:', data);
        Swal.fire({
            icon: 'success',
            title: 'Update Successful!',
            confirmButtonText: 'OK'
        });
    })
    .catch(error => {
        console.error('Error updating comment:', error);
    });
}

// แสดงกราฟโดยใช้ Chart.js
function displayResults(voteResult) {
    renderChart('vote-chart', voteResult, 'Vote Result');
}

// ฟังก์ชันสำหรับแสดงกราฟ
function renderChart(canvasId, data, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // ลบกราฟก่อนหน้าถ้ามี
    if (Chart.getChart(canvasId)) {
        Chart.getChart(canvasId).destroy();
    }

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: 'rgb(255,255,255)'
                    },
                    onClick: (e) => { }
                }
            }
        }
    });
}

//ฟังก์ชั่นแสดงสินค้าแว่น
function displayGlassesRecommendations(products) {
    const glassesContainer = document.getElementById('glasses-container');
    glassesContainer.innerHTML = '';

    products.forEach(product => {
        const glassesElement = document.createElement('div');
        glassesElement.className = "w-28 sm:w-32 lg:w-48 m-6 mx-3 flex flex-col self-start rounded-lg bg-white shadow-md transition duration-300 ease-in-out hover:scale-110";
        
        const buttonElement = document.createElement('button');
        buttonElement.setAttribute('data-te-toggle', 'modal');
        buttonElement.setAttribute('data-te-ripple-init', '');
        buttonElement.setAttribute('data-te-ripple-color', 'light');
    
        const imgElement = document.createElement('img');
        imgElement.src = product.glasses_img;
        imgElement.className = 'rounded-t-lg';
        
        buttonElement.appendChild(imgElement);
    
        const infoElement = document.createElement('div');
        infoElement.className = 'p-4 lg:p-6';
    
        glassesElement.appendChild(buttonElement);
        glassesElement.appendChild(infoElement);
    
        glassesContainer.appendChild(glassesElement);
    });
}
