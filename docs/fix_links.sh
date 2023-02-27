#! /bin/bash

# Fix broken links
# Create a list of all the files in the docs/docs/resources/bastionlab directory
find docs/docs/resources/bastionlab -type f -name "*.md" > list_path.txt
# Read the file 'list_path.txt' line by line
while read line; do
    # Read the entire file into an array
    IFS=$'\n' read -d '' -r -a lines < "$line"
    # Process each line in the array
    for line2 in "${lines[@]}"; do
        # Check if the line starts with '*'
        if [[ "$line2" == \** ]]; then
            # Remove the '* ' for the research of the pattern in the file
            line2=${line2:2}
            # Create the path of the file by replacing the '.' with '/'
            filepath=${line2//./\/}
            # Keep only the last part of the path
            filepath=${filepath##*/}
            # Check if filepath is 'psg'
            if [[ $filepath == 'psg' ]]; then
                # Check if it is just psg or torch.psg
                if [[ $line2 == 'bastionlab.torch.psg' ]]; then
                    sed -i "s+$line2+[$line2]($filepath/index.md)+g" $line
                fi
            elif [[ $filepath == 'pb' ]] || [[ $filepath == 'polars' ]] || [[ $filepath == 'torch' ]] || [[ $filepath == 'tokenizers' ]] ; then
                sed -i "s+$line2+[$line2]($filepath/index.md)+g" $line
            elif [[ $filepath == 'version' ]]; then
                # Delete the entire line as it is not needed
                sed -i "/$line2/d" $line
            else
                sed -i "s+\<$line2\>+[$line2]($filepath.md)+g" $line
            fi
        fi
    done
done < list_path.txt
rm list_path.txt
