import time
import requests

class DataDownloader:
    """
    Solved.ac data downloader
    """

    # Solved.ac main API url
    URL = 'https://solved.ac/api'

    def __init__(self, error_sleep_sec: int = 60, call_sleep_sec: int = 1) -> None:
        """Initialize downloader.

        Parameters
        ----------
        error_sleep_sec : int
            Sleeping seconds when getting too many requests error
        call_sleep_sec : int
            Sleeping seconds after calling API
        """
        self.error_sleep_sec = error_sleep_sec
        self.call_sleep_sec = call_sleep_sec

        # Max page per one API call
        # Set for preventing infinite loop just in case
        self._max_page = 100
        # Max problems when requesting multiple problem information
        # Must be set with caution - may blow up the server!
        self._max_problems = 100

    
    def _get(self, *args, **kwargs):
        """Wrapper function for requests.get()

        When it gets 429 error, sleep for a while and retry it.
        After getting response, sleep for a short time.

        Returns
        -------
        Any
            Return value of requests.get() with the parameter given.
        """
        
        response = requests.get(*args, **kwargs)
        time.sleep(self.call_sleep_sec)
        try:
            response.raise_for_status()
        except Exception as error:
            if response.status_code == 429:
                time.sleep(self.error_sleep_sec)
                response = self._get(*args, **kwargs)
            else:
                raise error
        return response

    def get_universities(self) -> list[dict]:
        """Get info about universities.

        Universities are sorted in an ascending order with respect to rank.

        Returns
        -------
        list[dict]
            List of all university information.
            [{'organizationId': ..., ...}, ...]
        """
        endpoint = '/v3/ranking/organization' 
        universities = []
        for cur_page in range(1, self._max_page + 1):
            params = {'type': 'university', 'page': cur_page}
            response = self._get(DataDownloader.URL + endpoint, params)
            if not response.json()['items']:
                break
            universities += response.json()['items']
        else:
            raise Exception('Max page reached!')
        return universities

    def get_students(self, univ_id: int) -> list[dict]:
        """Get all info about students that are at the given university.

        Students are sorted in an ascending order with respect to rank.

        Parameters
        ----------
        univ_id : int
            University ID where the user is registered.

        Returns
        -------
        list[dict]
            List of users who are at the university.
            [{'handle': ..., 'bio': ..., ...} ...]
        """
        endpoint = '/v3/ranking/in_organization'
        students = []
        for cur_page in range(1, self._max_page + 1):
            params = {'organizationId': univ_id, 'page': cur_page}
            response = self._get(DataDownloader.URL + endpoint, params)
            if not response.json()['items']:
                break
            students += response.json()['items']
        else:
            raise Exception('Max page reached!')
        return students

    def get_top_100_problems(self, handle: str) -> list[dict]:
        """Get info about top 100 problems that are solved by the handle.

        Problems are sorted in an ascending order by level.

        Parameters
        ----------
        handle : str
            Handle where problems were solved by.

        Returns
        -------
        dict
            List of top 100 problems that were solved by the handle.
            [{'problemId': ..., 'level': ..., 'tags': {...}, ...}, ...]
        """
        endpoint = '/v3/user/top_100'
        params = {'handle': handle, 'x-solvedac-language': 'ko'}
        response = self._get(DataDownloader.URL + endpoint, params)
        return response.json()['items']
    
    def get_problem(self, problem_id: int) -> dict:
        """Get problem info corresponding to the given problem id.

        Parameters
        ----------
        problem_id : int
            Problem id that will be searched.

        Returns
        -------
        dict
            Searched problem info.
            {'problemId': ..., 'level': ..., 'tags': {...}, ...}
        """
        endpoint = '/v3/problem/show'
        params = {'problemId': problem_id}
        response = self._get(DataDownloader.URL + endpoint, params)
        return response.json()
    
    def get_problems(self, problem_ids: list[int]) -> list[dict]:
        """Get problem info corresponding to the given problem ids.
        
           Number of problems must be less than self.max_problems (default to 100)

        Parameters
        ----------
        problem_ids : list[int]
            List of problem ids that will be searched.

        Returns
        -------
        list[dict]
            List of searched problem info.
            [{'problemId': ..., 'level': ..., 'tags': {...}, ...}, ...]
        """
        if len(problem_ids) > self._max_problems:
            raise Exception(f'Too many problems! - cur: {len(problem_ids)} > max: {self._max_problems}')
        endpoint = '/v3/problem/lookup'
        problem_ids = [str(id) for id in problem_ids]
        params = {'problemIds': ','.join(problem_ids)}
        response = self._get(DataDownloader.URL + endpoint, params)
        return response.json()
    
