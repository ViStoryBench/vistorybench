#!/bin/bash

# Define all possible metrics
ALL_METRICS="--cids --csd_cross --csd_self --aesthetic --prompt_align --diversity"

# Check if enough parameters are provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <method> [metrics...]"
    echo "Example: $0 uno --cids --csd_cross"
    exit 1
fi

METHOD="$1"
shift  # Remove the first parameter (method name), the rest are metrics

# Initialize METRICS variable
METRICS=""

# Process metric parameters
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        METRICS="$ALL_METRICS"
        break
    else
        METRICS="$METRICS $arg"
    fi
done

# Execute corresponding commands based on method name
case "$METHOD" in
        
    "uno")
        # uno
        python bench_run.py $METRICS \
            --method 'uno'
        ;;
        
    "seedstory")
        # seedstory
        python bench_run.py $METRICS \
            --method 'seedstory'
        ;;
        
    "storygen")
        # storygen
        for model_mode in 'auto-regressive' 'mix' 'multi-image-condition'; do
        python bench_run.py $METRICS \
            --method 'storygen' \
            --model_mode "$model_mode"
        done
        ;;
        
    "storydiffusion")
        # storydiffusion
        for content_mode in 'Photomaker' 'original'; do
            for style_mode in '(No style)'; do
                python bench_run.py $METRICS \
                    --method 'storydiffusion' \
                    --style_mode "$style_mode" \
                    --content_mode "$content_mode"
            done
        done
        ;;
        
    "storyadapter")
        # storyadapter
        for content_mode in 'img_ref' 'text_only'; do
            for scale_stage in 'results_xl5' 'results_xl'; do 
                python bench_run.py $METRICS \
                    --method 'storyadapter' \
                    --scale_stage "$scale_stage" \
                    --content_mode "$content_mode"
            done
        done
        ;;
        
    "theatergen")
        # theatergen
        python bench_run.py $METRICS \
            --method 'theatergen'
        ;;
        
    "movieagent")
        # movieagent
        for model_type in 'SD-3'; do
            python bench_run.py $METRICS \
                --method 'movieagent' \
                --model_type "$model_type"
        done
        ;;
        
    "vlogger")
        # vlogger
        for content_mode in 'img_ref' 'text_only'; do
            python bench_run.py $METRICS \
                --method 'vlogger' \
                --content_mode "$content_mode"
        done
        ;;
        
    "animdirector")
        # animdirector
        for model_type in 'sd3'; do
            python bench_run.py $METRICS \
                --method 'animdirector' \
                --model_type "$model_type"
        done
        ;;
        
    "mmstoryagent")
        # mmstoryagent
        python bench_run.py $METRICS \
            --method 'mmstoryagent'
        ;;
    
    "gemini"|"gpt4o")
        python bench_run.py $METRICS \
            --method "$METHOD"
        ;;
        
    "moki"|"morphic_studio"|"bairimeng_ai"|"shenbimaliang"|"xunfeihuiying"|"doubao")
        python bench_run.py $METRICS \
            --method "$METHOD"
        ;;
        
    "naive_baseline")
        python bench_run.py $METRICS \
            --method 'naive_baseline'
        ;;

    *)
        python bench_run.py $METRICS \
            --method "$METHOD"
            
        echo "Unknown method: $METHOD (but executed)"
        echo "Pre-define Available methods:"
        echo "Image: uno, seedstory, storygen, storydiffusion, storyadapter, theatergen"
        echo "Video: movieagent, vlogger, animdirector, mmstoryagent"
        echo "Closed source: gemini, gpt4o"
        echo "Business: moki, bairimeng_ai, shenbimaliang, xunfeihuiying, doubao"
        echo "And any other custom method names"
        ;;
esac

