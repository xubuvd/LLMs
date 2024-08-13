import sys
sys.path.append(r"..")
from utils.util import load_set_from_txt
_PolityToxic_ = "utils/political_toxic.txt"


from flashtext import KeywordProcessor

class CheckToxicWords():
    def __init__(self):
        '''
        self.political_words_set = load_set_from_txt(_PolityPersons_)
        self.sex_words = load_set_from_txt(_Sex_words_)
        self.ad_words = load_set_from_txt(_Ad_words_)
        '''

        self.processor = KeywordProcessor()
        self.processor.add_keyword_from_file(_PolityToxic_)

        '''
        aa = keyword_processor.extract_keywords('周杰伦是歌星在吉林大路开演唱会，导演国内有冯小刚，苏有朋演的是五阿哥，他现在居住在北京')
        print(aa)
        运行结果：
        ['明星', '路名', '明星', '明星', '地名']
        '''

    def is_toxic_text(self,text,thresh_hold=1):
        res = self.processor.extract_keywords(text)
        # ["politician","badword","gumble","sex","ads","dirty"]
        if len(res) >= thresh_hold: return True,"_".join(res)
        return False,""

    def checking_political_words(self,text):
        pass

    def checking_sex_words(self,text):
        pass

    def checking_ad_words(self,text):
        pass



