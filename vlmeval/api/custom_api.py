from ..smp import *
import os
import sys
from .base import BaseAPI
import time


class GPTCustomAPI(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 2048,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ['high', 'low']
        self.img_detail = img_detail
        self.timeout = timeout

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        self.api_base = 'http://10.21.226.169:8110/generate'
        self.headers = self.get_headers()

        self.logger.info(f'Using API Base: {self.api_base}')


    def get_headers(self):
        ak = '57daa242199c09d8834786c17801'
        sk = '505f046cf987330a5831f45e71a4'
        timestr = str(int(round(time.time() * 1000)))
        token = ak + "/" + timestr + "/" + sk
        sign = hashlib.sha256(token.encode('utf-8')).hexdigest()
        auth = ak + "/" + timestr + "/" + sign
        headers = {
            "content-type": "application/json",
            "Authorization": auth
        }

        return headers
    
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs
    
    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=self.headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        time.sleep(0.5)
        while response.status_code != 200:
            response = requests.post(
            self.api_base,
            headers=self.headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response
