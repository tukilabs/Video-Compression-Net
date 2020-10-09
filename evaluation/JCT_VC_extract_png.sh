extract() {
ffmpeg -y -pix_fmt yuv420p -s $3*$4 -i $1 -vframes 100 $2 2>&1 | tail -2
}

extract $1 $2 $3 $4
