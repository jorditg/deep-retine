IFS=$'\n'
FILES="../train/*.jpeg"
TOTAL=$(ls $FILES | wc -l)

i=0
for file in $(ls $FILES); do
	FILE=$file
	NAME=`echo "$FILE" | cut -d'/' -f3`

	convert $FILE -fill none -fuzz 5% -draw 'matte 0,0 floodfill' -flop -draw 'matte 0,0 floodfill' -flop $FILE-bck.jpeg
	./autotrim -f 5 -c NorthEast $FILE-bck.jpeg $NAME
	rm $FILE-bck.jpeg
	let i=i+1
        echo "$i of $TOTAL done - $file"
done

