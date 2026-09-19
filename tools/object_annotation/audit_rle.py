"""Standard-library COCO RLE codec for the portable human annotation editor."""


def validate_counts(counts,height,width):
    if not isinstance(counts,list) or not counts or len(counts)>height*width+1:
        raise ValueError('Invalid RLE run list')
    if any(type(n) is not int or n<0 for n in counts):
        raise ValueError('RLE counts must be nonnegative integers')
    if sum(counts)!=height*width:
        raise ValueError('RLE pixel count does not match image size')
    return sum(counts[1::2])


def encode_counts(counts,height,width):
    validate_counts(counts,height,width)
    encoded=[]
    for i,n in enumerate(counts):
        x=n-(counts[i-2] if i>2 else 0)
        while True:
            c=x&31;x>>=5
            more=x!=-1 if c&16 else x!=0
            if more:c|=32
            encoded.append(chr(c+48))
            if not more:break
    return dict(format='coco_rle',size=[height,width],counts=''.join(encoded))


def decode_counts(rle):
    if rle.get('format')!='coco_rle':raise ValueError('Expected COCO RLE')
    text=rle['counts'];counts=[];pos=0
    while pos<len(text):
        x=0;shift=0
        while True:
            if pos>=len(text) or shift>60:raise ValueError('Invalid compressed RLE')
            c=ord(text[pos])-48;pos+=1
            if c<0 or c>63:raise ValueError('Invalid RLE character')
            x|=(c&31)<<shift;shift+=5
            if not c&32:break
        if c&16:x|=-1<<shift
        if len(counts)>2:x+=counts[-2]
        counts.append(x)
    validate_counts(counts,*rle['size'])
    return counts
