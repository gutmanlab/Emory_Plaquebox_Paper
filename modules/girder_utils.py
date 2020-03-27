import girder_client

API_URLS = dict(
    CB='http://computablebrain.emory.edu:8080/api/v1',
)


def login(api_url=None, username=None, password=None, dsa=None):
    """Login to a girder client session.
    Parameters
    ----------
    api_url : str, optional
        DSA instance to use (hint: url ends with api/v1 most of the time), will be ignored if dsa is not None
    username : str, optional
        if both username and password are given, then client is authenticated non-interactively
    password : str, optional
        if both username and password are given, then client is authenticated non-interactively
    dsa : str, optional
        alternative to the api_url parameters, pass in CB for computablebrain, Transplant for transplant, candygram for
        candygram
    Returns
    -------
    gc : girder_client.GirderClient
        authenticated instance
    """
    if dsa is not None:
        try:
            api_url = API_URLS[dsa]
        except KeyError:
            raise Exception('dsa key not found: {}'.format(dsa))
    elif api_url is None:
        raise Exception("api_url and dsa parameters can't both be None")

    gc = girder_client.GirderClient(apiUrl=api_url)

    if username is not None and password is not None:
        gc.authenticate(username=username, password=password)
    else:
        gc.authenticate(interactive=True)
    return gc
