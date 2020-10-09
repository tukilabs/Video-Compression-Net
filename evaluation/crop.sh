crop() {
ffmpeg -pix_fmt yuv420p -s $3*$4 -i $1 -filter:v "crop=${5}:${6}:0:0" $2 2>&1 | tail -2
}

crop $1 $2 $3 $4 $5 $6
