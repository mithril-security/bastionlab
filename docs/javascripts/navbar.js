$(document).ready(function(){
    $('.md-nav[data-md-level="3"]').hide(); // hide all elements with class "md-nav" and data-md-level="3"
    $('.md-nav__item').click(function(){
        $(this).find('.md-nav[data-md-level="4"]').toggle(); // toggle all elements with class "md-nav" and data-md-level="4" inside the clicked element
    });
});