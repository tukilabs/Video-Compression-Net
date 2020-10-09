H264() {
ffmpeg -y -pix_fmt yuv420p -s $3*$4 -r $5 -i $1 -c:v libx264 -preset veryfast -tune zerolatency -psnr -crf $6 -g $7 -bf 2 -b_strategy 0 -sc_threshold 0 -loglevel debug $2 2>&1 | grep 'Global' | tail -1
}

echo "H264 CRF = ${6}"
H264 $1 $2 $3 $4 $5 $6 $7
