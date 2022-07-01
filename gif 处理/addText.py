# 参考：https://blog.csdn.net/weixin_39610415/article/details/110765583?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7-110765583-blog-112944306.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Edefault-7-110765583-blog-112944306.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=10

from PIL import Image,ImageFont,ImageDraw,ImageSequence

def watermark_on_gif(in_gif,out_gif,text='AAC最后10次迭代'):
    frames = []
    myfont = ImageFont.truetype("simkai.ttf", 20) # 加载字体对象
    im = Image.open(in_gif) # 打开gif图形
    water_im = Image.new("RGBA", im.size) # 新建RGBA模式的水印图
    draw = ImageDraw.Draw(water_im) # 新建绘画层
    draw.text(( 10, 10), text, font=myfont,fill='black')
    for frame in ImageSequence.Iterator(im): # 迭代每一帧
        frame = frame.convert("RGBA") # 转换成RGBA模式
        frame.paste(water_im,None,water_im) # 把水印粘贴到frame
        frames.append(frame) # 加到列表中
    newgif = frames[0] # 第一帧
    # quality参数为质量，duration为每幅图像播放的毫秒时间
    newgif.save(out_gif, save_all=True,
    append_images=frames[1:],
    quality=85,duration=100)
    im.close()

def main():
    watermark_on_gif("D:\\190-200-result.gif","D:\\190-200-AAC.gif")

if __name__ == "__main__":
    main()