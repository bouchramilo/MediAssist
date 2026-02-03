import pytest
from unittest.mock import patch, MagicMock
import os

from app.services.pdf_loader import load_pdf

MODULE_PATH = "app.services.pdf_loader"

@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.PyPDFDirectoryLoader")
def test_load_pdf_success(mock_loader_class, mock_exists):
    
    mock_exists.return_value = True
    
    mock_loader_instance = mock_loader_class.return_value
    mock_loader_instance.load.return_value = [
        MagicMock(page_content="Contenu de la page 1"),
        MagicMock(page_content="Contenu de la page 2")
    ]
    
    result = load_pdf()
    
    assert len(result) == 2
    assert result[0].page_content == "Contenu de la page 1"
    assert result[1].page_content == "Contenu de la page 2"
    
    mock_loader_class.assert_called_once()
    mock_loader_instance.load.assert_called_once()
    

@patch(f"{MODULE_PATH}.os.path.exists")
def test_load_pdf_directory_not_found(mock_exists):
    
    mock_exists.return_value = False
    
    result = load_pdf()
    
    assert result == []
    

@patch(f"{MODULE_PATH}.os.path.exists")
@patch(f"{MODULE_PATH}.PyPDFDirectoryLoader")
def test_load_pdf_exception(mock_loader_class, mock_exists):
    
    mock_exists.return_value == True
    
    mock_loader_instance = mock_loader_class.return_value
    mock_loader_instance.load.side_effect = Exception("Erreur de lecture")
    
    result = load_pdf()
    
    assert result == []
    
    