ffmpeg -ss 10.0 -t 10.0 -i rackoon_crawling.mp4 -filter_complex "[0:v] fps=12,scale=600:-1,split [a][b];[a] palettegen [p];[b][p] paletteuse" gifs/rackoon.gif -y

