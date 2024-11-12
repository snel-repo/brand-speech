# force using CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from brand import BRANDNode
import logging
import signal
import gc
from glob import glob
import sys
from deepmultilingualpunctuation import PunctuationModel
import stanza
import re

stanza.download('en')


class auto_punctuator(BRANDNode):
    def __init__(self):
        super().__init__()

        ## Load parameters, using `self.parameters`.
        self.input_stream = self.parameters.get('input_stream', 'text_for_punctuation')
        self.output_stream = self.parameters.get('output_stream', 'punctuated_text')
        self.capitalize = self.parameters.get('capitalize', False)

        self.last_input_entry_seen = self.get_current_redis_time_ms()

        self.punctuation_model = PunctuationModel()

        self.capitalization_model = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner', use_gpu=False)


    def get_current_redis_time_ms(self):
        t = self.r.time()
        return int(t[0]*1000 + t[1]/1000)
    

    def run(self):

        logging.info(f'{self.NAME} is ready to go')
        
        while True:

            read_result = self.r.xread(
                {self.input_stream: self.last_input_entry_seen},
                count = 1,
                block = 10 # ms
            )

            # timeout if no data received for X ms
            if len(read_result) == 0:
                continue

            # read this data snippet
            for entry_id, entry_dict in read_result[0][1]:
                self.last_input_entry_seen = entry_id
                text_for_punctuation = entry_dict[b'text_for_punctuation'].decode()

            try:
                if self.capitalize:
                    capitalized = self.capitalization_model(text_for_punctuation)
                    capitalized = [
                        word.text.capitalize() 
                            if word.upos in ["PROPN", "NNS"]
                            else word.text
                        for sentence in capitalized.sentences for word in sentence.words]
                    capitalized = [word.capitalize() if word == 'i' else word for word in capitalized]
                    capitalized[0] = capitalized[0].capitalize()
                    text_for_punctuation = re.sub(" (?=[\.,'!?:;]|n't)", "", ' '.join(capitalized))
                punctuated_text = self.punctuation_model.restore_punctuation(text_for_punctuation)
            except:
                punctuated_text = text_for_punctuation

            self.r.xadd(self.output_stream, {'punctuated_text': punctuated_text})

    def terminate(self, sig, frame):
        logging.info('SIGINT received, Exiting')
        gc.collect()
        sys.exit(0)

if __name__ == "__main__":
    gc.disable()

    node = auto_punctuator()
    node.run()

    gc.collect()