
import urllib.request
import logging
import threading
import time

logger = logging.getLogger(__name__)



def api_call(url :str ) ->str :
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request) as r:
        body = r.read().decode(r.info().get_param("charset") or "utf-8")
    return body


def api_call_retry(url :str , numTries :int, delay : int = 0 ) ->str :

    for attempt in range(1, numTries):
    
        try:
            return api_call( url )
        
        except Exception as e:   
            
            if attempt<numTries :
                err_msg = f"{e.__class__.__name__}: {e}"
                logger.warning("T%d:Exception in api_call_retry(%s) url=%s", threading.get_ident(), err_msg, url)
                
                if delay>0:
                    time.sleep( delay )
                    
                logger.warning("T%d:Trying again, attempt %d of %d.", threading.get_ident(), attempt+1, numTries)
            else:
                raise
