// Button Click Animation
document.querySelector('.btn-register').addEventListener('click', () => {
    alert('Thank you for registering with Safe Frame!');
});

// Smooth Scroll for Navigation Links
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', event => {
        event.preventDefault();
        const targetId = event.target.getAttribute('href').slice(1);
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 50,
                behavior: 'smooth',
            });
        }
    });
});

// Image Hover Effect (Optional Interactive Animation)
document.querySelectorAll('.content-section img').forEach(image => {
    image.addEventListener('mouseenter', () => {
        image.style.transform = 'rotate(3deg) scale(1.1)';
        image.style.transition = 'transform 0.5s ease';
    });
    image.addEventListener('mouseleave', () => {
        image.style.transform = 'rotate(0) scale(1)';
        image.style.transition = 'transform 0.5s ease';
    });
});

// Smooth Scroll for Learn More Button to About Section
document.querySelector('.btn-learn-more').addEventListener('click', () => {
    const aboutSection = document.getElementById('about');
    window.scrollTo({
        top: aboutSection.offsetTop - 50,
        behavior: 'smooth',
    });
});

// Image Hover Effect (Optional Interactive Animation)
document.querySelectorAll('.about-image img').forEach(image => {
    image.addEventListener('mouseenter', () => {
        image.style.transform = 'rotate(3deg) scale(1.1)';
        image.style.transition = 'transform 0.5s ease';
    });
    image.addEventListener('mouseleave', () => {
        image.style.transform = 'rotate(0) scale(1)';
        image.style.transition = 'transform 0.5s ease';
    });
});

// Smooth scrolling to services section when needed
document.querySelectorAll('a[href^="#services"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    document.querySelector(this.getAttribute('href')).scrollIntoView({
      behavior: 'smooth'
    });
  });
});


// Contact form submission event
document.querySelector('.contact-form').addEventListener('submit', function (e) {
  e.preventDefault();
  alert('Your message has been sent successfully!');
  // Reset form fields
  this.reset();
});
