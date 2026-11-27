#!/usr/bin/env python3
"""Render the accompanying Markdown subset into a Chinese research PDF."""
from __future__ import annotations
import argparse, html, re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

def inline(text):
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:'<link href="'+m.group(2)+'" color="#1C4D72">'+m.group(1)+'</link>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    return text

def render(source,target,fontdir):
    pdfmetrics.registerFont(TTFont('NotoSC',str(fontdir/'NotoSansSC-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('NotoSCBold',str(fontdir/'NotoSansSC-Bold.ttf')))
    pdfmetrics.registerFontFamily('NotoSC',normal='NotoSC',bold='NotoSCBold',italic='NotoSC',boldItalic='NotoSCBold')
    styles={
        'p':ParagraphStyle('Body',fontName='NotoSC',fontSize=9.7,leading=15.3,spaceAfter=7.5,wordWrap='CJK',allowWidows=0,allowOrphans=0),
        'h1':ParagraphStyle('Title',fontName='NotoSCBold',fontSize=20,leading=28,spaceAfter=15,keepWithNext=True,wordWrap='CJK'),
        'h2':ParagraphStyle('Section',fontName='NotoSCBold',fontSize=13.2,leading=20,spaceBefore=13,spaceAfter=9,keepWithNext=True,wordWrap='CJK'),
        'h3':ParagraphStyle('Subsection',fontName='NotoSCBold',fontSize=11,leading=17,spaceBefore=10,spaceAfter=7,keepWithNext=True,wordWrap='CJK'),
        'cell':ParagraphStyle('Cell',fontName='NotoSC',fontSize=8.4,leading=13.2,wordWrap='CJK',spaceAfter=0),
        'th':ParagraphStyle('Header',fontName='NotoSCBold',fontSize=8.4,leading=13.2,wordWrap='CJK',spaceAfter=0),
        'caption':ParagraphStyle('Caption',fontName='NotoSC',fontSize=8.6,leading=13,spaceBefore=4,spaceAfter=10,wordWrap='CJK'),
        'ref':ParagraphStyle('Reference',fontName='NotoSC',fontSize=8.6,leading=13.7,spaceAfter=9,wordWrap='CJK'),
    }
    width=A4[0]-88
    story=[];lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('!['):
            m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
            p=(source.parent/m.group(2)).resolve()
            im=Image(str(p));im.drawHeight=im.imageHeight*width/im.imageWidth;im.drawWidth=width
            story.append(KeepTogether([im,Paragraph(inline(m.group(1)),styles['caption'])]));i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):table.append(cells)
                i+=1
            n=len(table[0]);assert all(len(row)==n for row in table)
            headers=table[0]
            if n==7:ratios=[1.05,.50,1.1,1.1,1.1,.8,.8]
            elif n==6:ratios=[.65,.8,1,1,1,1]
            elif n==5:ratios=[1.6,1.1,1,1.1,1]
            elif n==4:ratios=[1.05,1.65,1.5,1.6]
            else:ratios=[1.1]+[2.0]*(n-1)
            if headers[:2]==['通道 w/2w','参数量']:ratios=[1.3,1,1.2,1.3]
            if headers[:2]==['结构','通道']:ratios=[2.1,1,.9,1.1,1]
            if headers[:2]==['预算','CE']:ratios=[.7,1.25,1.25,1.25,1,1]
            if n==5 and headers[0]=='预算':ratios=[.5,1.35,1.35,1.15,1.35]
            widths=[width*r/sum(ratios) for r in ratios]
            data=[[Paragraph(inline(c),styles['th' if r==0 else 'cell']) for c in row] for r,row in enumerate(table)]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            ts=[('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF4')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]
            for r in range(1,len(data)):
                if r%2==0:ts.append(('BACKGROUND',(0,r),(-1,r),colors.HexColor('#F6F8FA')))
            t.setStyle(TableStyle(ts));story.extend([t,Spacer(1,9)]);continue
        if line.startswith('# '):style='h1';text=line[2:]
        elif line.startswith('## '):style='h2';text=line[3:]
        elif line.startswith('### '):style='h3';text=line[4:]
        else:
            style='ref' if re.match(r'^\[\d+\]',line) else 'p';text=line
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].lstrip().startswith(('#','|','![')):
                i+=1;text+=' '+lines[i].strip()
        story.append(Paragraph(inline(text),styles[style]));i+=1
    def page(canvas,doc):
        canvas.saveState();canvas.setFont('NotoSC',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(44,24,'1D CNN HAR 研究进展  2026-11-26')
        canvas.drawRightString(A4[0]-44,24,f'{doc.page}')
        canvas.restoreState()
    target.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=38,bottomMargin=43,
        title='1D CNN 人体活动识别蒸馏研究进展',author='研究辅助报告')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(target)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('target',type=Path);ap.add_argument('fontdir',type=Path)
    a=ap.parse_args();render(a.source,a.target,a.fontdir)
