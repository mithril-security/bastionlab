var navbar_ = document.getElementsByClassName('md-sidebar md-sidebar--primary')[0].children[0];

navbar_.addEventListener('scroll', (event) => {

    lastKnownScrollPosition = navbar_.scrollTop;
    // Create a temporary file to store the last known scroll position
    // Save a value to localStorage
    localStorage.setItem("navbar_position", lastKnownScrollPosition);
    console.log("Saved scroll position: " + lastKnownScrollPosition);
}
);

try {
     // Retrieve the value from localStorage
    let navbar_position = localStorage.getItem("navbar_position");
    console.log("Navbar was at pos" + navbar_position); // "John Doe"
    navbar_.scrollTop = navbar_position;
}
catch (e) {
    console.log("unable to fetch latest scroll position")
}
