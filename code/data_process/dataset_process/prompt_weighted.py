
import sys
code_path = '/data/AIGC_Research/Story_Telling/StoryVisBMK/code'
sys.path.append(f'{code_path}/data_process/dataset_process/sd_embed/src/sd_embed')

def prompt_weighted_encode(pipe, prompt, neg_prompt, model_type,**kwargs):
    import gc

    # prompt = prompt[0]
    # neg_prompt = neg_prompt[0]

    if model_type == 'sd15':
        from embedding_funcs import get_weighted_text_embeddings_sd15
        (
            prompt_embeds,
            prompt_neg_embeds
        ) = get_weighted_text_embeddings_sd15(
            pipe,
            prompt = prompt,
            neg_prompt = neg_prompt,
            **kwargs
        )
        return (
            prompt_embeds,
            prompt_neg_embeds
        )

    elif model_type == 'sdxl':
        print("--- Before get_weighted_text_embeddings_sdxl ---")
        print(prompt)
        print(neg_prompt)
        print("--- After get_weighted_text_embeddings_sdxl ---")
        from embedding_funcs import get_weighted_text_embeddings_sdxl
        (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = get_weighted_text_embeddings_sdxl(
            pipe,
            prompt = prompt,
            neg_prompt = neg_prompt,
            **kwargs
        )
        return (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) 

    elif model_type == 'sd3':
        from embedding_funcs import get_weighted_text_embeddings_sd3
        (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = get_weighted_text_embeddings_sd3(
            pipe,
            prompt = prompt,
            neg_prompt = neg_prompt,
            **kwargs
        )
        return (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        )

    elif model_type == 'storyadapterxl':
        from embedding_funcs import get_weighted_text_embeddings_storyadapterxl
        # 在StoryAdapterXL的generate方法中替换encode_prompt调用
        (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = get_weighted_text_embeddings_storyadapterxl(
            pipe,
            prompt = prompt,
            neg_prompt = neg_prompt,
            **kwargs
        )
        return (
            prompt_embeds,
            prompt_neg_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        )

def main():

    pipe = ''
    prompt = ''
    neg_prompt = ''

    (
        prompt_embeds, 
        prompt_neg_embeds
    ) = prompt_weighted_encode(
        pipe, 
        prompt, 
        neg_prompt, 
        'sd15'
    )

    (
        prompt_embeds, 
        prompt_neg_embeds, 
        pooled_prompt_embeds, 
        negative_pooled_prompt_embeds
    ) = prompt_weighted_encode(
        pipe, 
        prompt, 
        neg_prompt,
        'sdxl'
    )
    
    (
        prompt_embeds, 
        prompt_neg_embeds, 
        pooled_prompt_embeds, 
        negative_pooled_prompt_embeds
    ) = prompt_weighted_encode(
        pipe, 
        prompt, 
        neg_prompt,
        'sd3'
    )


def example_for_sd_15():

    import gc
    import torch
    from diffusers import StableDiffusionPipeline
    from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15

    model_path = "stablediffusionapi/deliberate-v2"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )

    pipe.to('cuda')

    prompt = """A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
    This imaginative creature features the distinctive, bulky body of a hippo, 
    but with a texture and appearance resembling a golden-brown, crispy waffle. 
    The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
    It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
    possibly including oversized utensils or plates in the background. 
    The image should evoke a sense of playful absurdity and culinary fantasy.
    """

    neg_prompt = """\
    skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
    (tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
    extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
    (too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
    bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
    (normal quality:2),lowres,((monochrome)),((grayscale))
    """

    (
        prompt_embeds
        , prompt_neg_embeds
    ) = get_weighted_text_embeddings_sd15(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )

    image = pipe(
        prompt_embeds                   = prompt_embeds
        , negative_prompt_embeds        = prompt_neg_embeds
        , num_inference_steps           = 30
        , height                        = 768
        , width                         = 896
        , guidance_scale                = 8.0
        , generator                     = torch.Generator("cuda").manual_seed(2)
    ).images[0]
    display(image)

    del prompt_embeds, prompt_neg_embeds
    pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()





def example_for_sdxl():


    import gc
    import torch
    from diffusers import StableDiffusionXLPipeline
    from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl

    model_path = "Lykon/dreamshaper-xl-1-0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16
    )
    pipe.to('cuda')

    prompt = """A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. 
    This imaginative creature features the distinctive, bulky body of a hippo, 
    but with a texture and appearance resembling a golden-brown, crispy waffle. 
    The creature might have elements like waffle squares across its skin and a syrup-like sheen. 
    It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, 
    possibly including oversized utensils or plates in the background. 
    The image should evoke a sense of playful absurdity and culinary fantasy.
    """

    neg_prompt = """\
    skin spots,acnes,skin blemishes,age spot,(ugly:1.2),(duplicate:1.2),(morbid:1.21),(mutilated:1.2),\
    (tranny:1.2),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.2),(bad proportions:1.3),\
    extra limbs,(disfigured:1.2),(missing arms:1.2),(extra legs:1.2),(fused fingers:1.5),\
    (too many fingers:1.5),(unclear eyes:1.2),lowers,bad hands,missing fingers,extra digit,\
    bad hands,missing fingers,(extra arms and legs),(worst quality:2),(low quality:2),\
    (normal quality:2),lowres,((monochrome)),((grayscale))
    """

    (
        prompt_embeds
        , prompt_neg_embeds
        , pooled_prompt_embeds
        , negative_pooled_prompt_embeds
    ) = get_weighted_text_embeddings_sdxl(
        pipe
        , prompt = prompt
        , neg_prompt = neg_prompt
    )

    image = pipe(
        prompt_embeds                   = prompt_embeds
        , negative_prompt_embeds        = prompt_neg_embeds
        , pooled_prompt_embeds          = pooled_prompt_embeds
        , negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        , num_inference_steps           = 30
        , height                        = 1024 
        , width                         = 1024 + 512
        , guidance_scale                = 4.0
        , generator                     = torch.Generator("cuda").manual_seed(2)
    ).images[0]
    display(image)

    del prompt_embeds, prompt_neg_embeds,pooled_prompt_embeds, negative_pooled_prompt_embeds
    pipe.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()