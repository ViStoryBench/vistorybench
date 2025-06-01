
#!/bin/bash

# 定义所有可能的指标
ALL_METRICS="--cref --csd_cross --csd_self --aesthetic --prompt_align2 --diversity"

# 检查是否提供了足够的参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <method> [metrics...]"
    echo "Example: $0 uno --cref --csd_cross"
    exit 1
fi

METHOD="$1"
shift  # 移除第一个参数（方法名），剩下的都是指标

# 初始化METRICS变量
METRICS=""

# 处理指标参数
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        METRICS="$ALL_METRICS"
        break
    else
        METRICS="$METRICS $arg"
    fi
done

# 根据方法名执行相应的命令
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
        for model_mode in 'multi-image-condition'; do
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
        for content_mode in 'text_only'; do
            for scale_stage in 'results_xl5'; do 
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
        echo "Unknown method: $METHOD"
        echo "Available methods:"
        echo "Image: uno, seedstory, storygen, storydiffusion, storyadapter, theatergen"
        echo "Video: movieagent, vlogger, animdirector, mmstoryagent"
        echo "Closed source: gemini, gpt4o"
        echo "Business: moki, bairimeng_ai, shenbimaliang, xunfeihuiying, doubao"
        exit 1
        ;;
esac




# # /////////// Closed source ///////////
# for method in 'gemini' 'gpt4o'; do
#     python bench_run.py $METRICS \
#         --method "$method"
# done

# /////////// Business methods ///////////
# # for method in 'moki' 'morphic_studio' 'bairimeng_ai' 'shenbimaliang' 'xunfeihuiying' 'doubao'; do
# # morphic_studio数据未完整，暂时先排除
# for method in 'moki' 'bairimeng_ai' 'shenbimaliang' 'xunfeihuiying' 'doubao'; do
#     python bench_run.py $METRICS \
#         --method "$method"
# done


