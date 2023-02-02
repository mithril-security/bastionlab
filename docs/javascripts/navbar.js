$(document).ready(function () {
    $('.md-nav__link').each(function () {
        var subMenu = $(this).parent().find('.md-nav[data-md-level="2"]');
        if (subMenu.length) {
            subMenu.show();
            $(this).find('.md-nav__icon').css('transform', 'rotate(90deg)');
            // Add a click event to the link
            $(this).click(function () {
                // Fing the sub-menu element
                var subMenu = $(this).parent().find('.md-nav[data-md-level="2"]');
                // If the sub-menu is hidden, show it and rotate the arrow-z
                if (subMenu.is(':hidden')) {
                    subMenu.show();
                    // Rotate the arom
                    $(this).find('.md-nav__icon').css('transform', 'rotate(27deg)');
                } else {
                    // Hide the sub-menu and rotate the arrow
                    subMenu.hide();
                    $(this).find('.md-nav__icon').css('transform', 'rotate(-27deg)');
                }
            });
        }
    });
});


// $(document).ready(function () {
//     // Find every nav with class "md-nav"
//     $('.md-nav').each(function () {
//         // Find every nav item with class ""