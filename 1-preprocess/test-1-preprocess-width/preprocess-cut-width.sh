IFS=$'\n'
DIR="../../0-downloads/test/"
TOTAL=$(find $DIR -name *.jpeg | sed '1d' | wc -l)
VAR="10%"

i=0
for file in $(find $DIR -name *.jpeg | sed '1d'); do
	SRC=$file
	DST=$(basename $SRC)
	convert $SRC -fill none -fuzz $VAR -draw 'matte 0,0 floodfill' -flop -draw 'matte 0,0 floodfill' -flop bck-$DST
	./autotrim -f $VAR -c NorthEast bck-$DST $DST
	rm bck-$DST
	let i=i+1
        echo "$i of $TOTAL done - $file"
done

