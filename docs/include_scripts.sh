#! /bin/bash

# List all of the generated html files in the docs directory

find ./site -name "*.html" | for file in $(cat); do
    # Check in the file if there is already a script with src="javascripts/navbar_pos.js"
    # If not, add it to the end of the body tag
    if ! grep -q "navbar_pos.js" $file; then
        sed -i '/<\/body>/i \ <script src="javascripts/navbar_pos.js"></script>' $file
    fi
done

