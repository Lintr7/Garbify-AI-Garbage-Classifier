$(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
});


const navButtons = document.querySelectorAll('.nav-btn');
const videos = document.querySelectorAll('.video-background');

const defaultButton = navButtons[0];
defaultButton.classList.add('selected');
videos[0].classList.add('active');
videos[0].play()


function updateVideo(button) {
    const index = button.getAttribute('data-index');

    navButtons.forEach(btn => btn.classList.remove('selected'));

    button.classList.add('selected');

    videos.forEach(video => video.classList.remove('active'));

    videos[index].classList.add('active');

}


// Add event listeners for video loop reset
videos.forEach(video => {
    video.addEventListener('ended', () => handleVideoLoop(video));
});

// Add click event listeners to each button
navButtons.forEach(button => {
    button.addEventListener('click', () => updateVideo(button));
});