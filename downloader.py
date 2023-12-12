"""
Code to download Hi-C experiments from ENCODE database.
"""
from urllib.parse import urljoin
import os
import numpy as np
from wget import download
from wget import bar_thermometer
import requests

def get_json(url: str) -> dict:
    """get information from the url in json format.

    Args:
        url (str): url of the web page.

    Returns:
        dict: return json from the web page as a dictionary 
    """
    headers: dict = {'accept': 'application/json'}
    # GET the search result
    response = requests.get(url, headers=headers, timeout = 10)

    # Extract the JSON response as a python dictionary
    return response.json()

def search_encode_url(search_parameters: dict, search_filter: dict = None) -> str:
    """Create the search url given search parameters and filters.

    Args:
        search_parameters (dict): for the query of encode.
        search_filter (dict): for filtering out some of the search results.

    Returns:
        dict: return the search result as a dictionary. 
    """
    netloc = 'https://www.encodeproject.org/search/'

    query = '?' + '&'.join(["&".join([f"{k}={i}" for i in v]) if isinstance(v, list)
                            else f"{k}={v}"
                            for k, v in search_parameters.items()])
    if search_filter is not None:
        query += '&' + '&'.join(["&".join([f"{k}!={i}" for i in v]) if isinstance(v, list)
                                else f"{k}!={v}"
                                for k, v in search_filter.items()])

    return query

def search_encode(search_parameters: dict, search_filter: dict = None) -> dict:
    """Search encode using parameters and filters provided and return the resulting
    page in json format.

    Args:
        search_parameters (dict): for the query of encode.
        search_filter (dict): for filtering out some of the search results.

    Returns:
        dict: return the search result as a dictionary. 
    """
    netloc = 'https://www.encodeproject.org/search/'

    query = search_encode_url(search_parameters, search_filter)

    search_url = urljoin(netloc, query)

    return get_json(search_url)

def download_file(fid: str, out: str) -> None:
    """Download file from an encode experiment.

    Args:
        fid (str): encode file id.
        out (str): donwload directory.
    """
    print(f"Downloading file {fid}:")
    encode = 'https://www.encodeproject.org/'
    url = urljoin(encode, f"/files/{fid}/")
    page = get_json(url)
    download(urljoin(encode, page["href"]), out = out, bar = bar_thermometer)

def download_experiment(eid: str, main_folder: str = ".") -> None:
    """Download the preferred files from the released analyses of the experiment from encode.

    Args:
        id (str): encode experiment id. 
        main_folder (str, optional): main folder to download all files to. Defaults to ".".
    """
    print(f"Downloading experiment {eid}:")

    folder = os.path.join(main_folder, eid)

    if not os.path.isdir(folder):
        os.mkdir(folder)

    url = urljoin('https://www.encodeproject.org/', eid)
    page = get_json(url)
    analyses = np.array(page["analyses"])
    released = analyses[[i["status"] == "released" for i in analyses]]
    files = sum([i["files"] for i in released], start = [])
    files = list(filter(lambda x: f"/files/{x['accession']}/" in files and
                        "preferred_default" in x.keys() and x["preferred_default"],
                page["files"]))
    for file in files:
        download_file(file["accession"], folder)

def download_chipseq_experiment_fold_change_replicates(eid: str, main_folder: str = ".") -> None:
    """Download the files of fold change over control from the released analyses of the experiment
    from encode which have more than two replicates.

    Args:
        id (str): encode experiment id. 
        main_folder (str, optional): main folder to download all files to. Defaults to ".".
    """
    print(f"Downloading experiment {eid}:")

    folder = os.path.join(main_folder, eid)

    if not os.path.isdir(folder):
        os.mkdir(folder)

    url = urljoin('https://www.encodeproject.org/', eid)
    page = get_json(url)
    default_analysis = page["default_analysis"]
    for file in page["files"]:
        if default_analysis in [i["@id"] for i in file["analyses"]] and \
           file["output_type"] == "fold change over control" and \
           len(file["biological_replicates"]) >= 2 and \
           file["file_type"].lower() == "bigwig":
            download_file(file["accession"], os.path.join(folder, file["accession"] +
                                                       "_" + file["target"].split("/")[-2]) + \
                                                       "." + file["file_type"].lower())

def return_file_size_chipseq_experiment_fold_change_replicates(eid: str) -> None:
    """Return size of an experiment on encode.

    Args:
        id (str): encode experiment id. 
    """
    url = urljoin('https://www.encodeproject.org/', eid)
    page = get_json(url)
    default_analysis = page["default_analysis"]
    file_size_sum = 0
    for file in page["files"]:
        if default_analysis in [i["@id"] for i in file["analyses"]] and \
           file["output_type"] == "fold change over control" and \
           len(file["biological_replicates"]) >= 2:
            file_size_sum += file["file_size"]
    return file_size_sum

def return_data_size_chipseq_cell_line(cell_line: str, limit: int) -> int:
    """Return data size for ChIP seq files to be downloaded.

    Args:
        cell_line (str): target cell line
        limit (int): limit the number of experiments to be downloaded

    Returns:
        int: total data size in terms of bytes.
    """

    search_parameters = {
        "type": "Experiment",
        "biosample_ontology.term_name": cell_line,
        "replicates.library.biosample.donor.organism.scientific_name": "Homo+sapiens",
        "assembly": "GRCh38",
        "files.file_type": "bigWig",
        "status": "released",
        "assay_title": ["TF+ChIP-seq",
                        "Histone+ChIP-seq"],
        "limit": limit
    }

    search_filter = {
        "audit.ERROR.category": ["extremely+low+read+depth",
                                "control+extremely+low+read+depth",
                                "extremely+low+coverage",
                                "missing+control+alignments",
                                "missing+lambda+C+conversion+rate"],
        "audit.NOT_COMPLIANT.category": "unreplicated+experiment",
    }

    search_results = search_encode(search_parameters, search_filter)

    total_data_size = 0
    for experiment in search_results["@graph"]:
        experiment_id = experiment["accession"]
        total_data_size += return_file_size_chipseq_experiment_fold_change_replicates(experiment_id)
    return total_data_size

def download_hic_cell_line(cell_line: str, folder: str):
    """Download HiC experiments for the given cell line from encode. 

    Args:
        cell_line (str): the name of the cell line 
        folder (str): download folder.
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)
    search_parameters = {
        'type': 'Experiment',
        'assay_title': ['in+situ+Hi-C', 'Hi-C'],
        'biosample_ontology.classification': 'cell+line',
        'replicates.library.biosample.donor.organism.scientific_name': 'Homo+sapiens',
        'biosample_ontology.term_name': cell_line,
        'assembly': 'GRCh38',
        'status': 'released',
        'limit': '5'
    }
    search_results = search_encode(search_parameters)
    for experiment in search_results["@graph"]:
        experiment_id = experiment["accession"]
        download_experiment(experiment_id, folder)

def download_chipseq_cell_line(cell_line: str, folder: str, limit: int):
    """Download all ChIP-seq data from encode database.

    Args:
        cell_line (str): target cell line
        folder (str): download folder 
        limit (int): limit the number of experiments
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    search_parameters = {
        "type": "Experiment",
        "biosample_ontology.term_name": cell_line,
        "replicates.library.biosample.donor.organism.scientific_name": "Homo+sapiens",
        "assembly": "GRCh38",
        "files.file_type": "bigWig",
        "status": "released",
        "assay_title": ["TF+ChIP-seq",
                        "Histone+ChIP-seq"],
        "limit": limit
    }

    search_filter = {
        "audit.ERROR.category": ["extremely+low+read+depth",
                                "control+extremely+low+read+depth",
                                "extremely+low+coverage",
                                "missing+control+alignments",
                                "missing+lambda+C+conversion+rate"],
        "audit.NOT_COMPLIANT.category": "unreplicated+experiment",
    }

    search_results = search_encode(search_parameters, search_filter)

    for experiment in search_results["@graph"]:
        experiment_id = experiment["accession"]
        download_chipseq_experiment_fold_change_replicates(experiment_id, folder)
