var navbar = document.getElementsByClassName('md-nav')[0];
var menu = navbar.querySelectorAll('[data-md-level="1"]');


var submenus = [];
var menu_length = menu.length;
var arrows = [];

// Iterate on the submenus level 1
for (var i = 0; i < menu_length; i++) {
    var to_add = []
    if (menu[i].querySelectorAll('[data-md-level="2"]').length > 0) {
        // Iterate over the children of the ul element which is the children in position 1 of the menu element
        for (var j = 0; j < menu[i].children[1].childNodes.length; j++) {
            // Check if the element is a li element
            if (menu[i].children[1].childNodes[j].tagName == 'LI') {
                to_add.push(menu[i].children[1].childNodes[j]);
                // Iterate over the children of the li element to find the element with the class md-nav__link  
                for (var k = 0; k < menu[i].children[1].childNodes[j].childNodes.length; k++) {
                    // Check if the element is a div element
                    if (menu[i].children[1].childNodes[j].childNodes[k].tagName == 'DIV') {
                        // Check if the element has the class md-nav__link
                        if (menu[i].children[1].childNodes[j].childNodes[k].classList.contains('md-nav__link')) {
                            arrow = menu[i].children[1].childNodes[j].childNodes[k].children[1].children[0];
                            arrows.push(arrow);
                        }
                    }
                    
                    if (menu[i].children[1].childNodes[j].childNodes[k] !== undefined &&
                        menu[i].children[1].childNodes[j].childNodes[k] !== null &&
                        menu[i].children[1].childNodes[j].childNodes[k].classList !== undefined &&
                        menu[i].children[1].childNodes[j].childNodes[k].classList !== null &&
                        menu[i].children[1].childNodes[j].childNodes[k].classList.contains('md-nav__link')) {
                        for (var l = 0; l < menu[i].children[1].childNodes[j].childNodes[k].childNodes.length; l++) {
                            var arrow = menu[i].children[1].childNodes[j].childNodes[k].childNodes[l];
                            if (arrow != undefined && arrow.classList != undefined && arrow.classList.contains('md-nav__icon')) {
                                arrows.push(arrow);
                            }
                        }
                    }
                }
            }
        }
        submenus.push(to_add);
    }
}

var i_arrow = 0;

for (var i = 0; i < submenus.length; i++) {
    for (var j = 0; j < submenus[i].length; j++) {
        for (var k = 0; k < submenus[i][j].childNodes.length; k++) {
            if (submenus[i][j].childNodes[k].tagName == 'NAV') {
                submenus[i][j].childNodes[k].style.display = 'block';
                var submenus_nested = submenus[i][j].childNodes[k];

                // Add a id attribute to the submenu with the id of the arrow
                submenus_nested.setAttribute('id', i_arrow);
                arrows[i_arrow].setAttribute('for', i_arrow);

                // rotate the arrow by 90 degrees
                arrows[i_arrow].style.transform = 'rotate(90deg)';
                // Add event listener to the arrow
                arrows[i_arrow].addEventListener('click', function () {

                    // Check the position of the arrow
                    if (this.style.transform == 'rotate(90deg)') {
                        // rotate the arrow by 90 degrees
                        this.style.transform = 'rotate(0deg)';
                    }
                    else {
                        // rotate the arrow by 90 degrees
                        this.style.transform = 'rotate(90deg)';
                    }

                    // Look for the submenu with the for attribute equal to the id of the arrow
                    var submenus_corresponding = document.getElementById(this.getAttribute('for'));
                    if (submenus_corresponding.style.display == 'block') {
                        submenus_corresponding.style.display = 'none';
                    }
                    else {
                        submenus_corresponding.style.display = 'block';
                    }
                }
                );
            }
        }
        i_arrow++;
    }
}
