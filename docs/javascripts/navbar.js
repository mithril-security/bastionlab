// Search for every .md-nav__link element
// For every .md-nav__link element, verify if it contains an .md-nav[data-md-level="2"] element
// If it contains an .md-nav[data-md-level="2"] element, display it and rotate the arrow by 90 degrees
// If it doesn't contain an .md-nav[data-md-level="2"] element, do not touch it and do not rotate the arrow

$(document).ready(function () {
    $('.md-nav__link').each(function () {
        var subMenu = $(this).parent().find('.md-nav[data-md-level="2"]');
        if (subMenu.length) {
            subMenu.show();
            $(this).find('.md-nav__icon').css('transform', 'rotate(90deg)');
            // Add a click event to the .md-nav__link element
            $(this).click(function () {
                // Find the sub-menu element
                var subMenu = $(this).parent().find('.md-nav[data-md-level="2"]');
                // If the sub-menu is hidden, show it
                if (subMenu.is(':hidden')) {
                    subMenu.show();
                    // Rotate the arrow down
                    // $(this).find('.md-nav__icon').css('transform', 'rotate(90deg)');
                }
                // If the sub-menu is visible, hide it
                else {
                    subMenu.hide();
                    // Rotate the arrow by -90 degrees
                    // $(this).find('.md-nav__icon').css('transform', 'rotate(-90deg)');
                }
            }
            );
        }
    });
});