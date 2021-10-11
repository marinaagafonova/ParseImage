import fitz
import io
from PIL import Image


def extract_images_from_pdf(filename):
    pdf_file = fitz.open(filename)
    # проход по страницам pdf-файла
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index] # получение конкретной страницы
        image_list = page.getImageList()
        
        # вывод о результате парсинга
        if image_list:
            print(f"[+] Количество найденных изображений на странице {page_index} : {len(image_list)}")
        else:
            print("[!] Не было найдено изображений на странице ", page_index)

        for image_index, img in enumerate(image_list, start=1):
            
            xref = img[0] # получение внешней ссылки (XREF) изображения

            # ивлечение байтов изображения
            base_image = pdf_file.extractImage(xref)
            image_bytes = base_image["image"]
            
            image_ext = base_image["ext"] # получение расширения изображения

            image = Image.open(io.BytesIO(image_bytes)) # загружка в PIL
            image.save(open(f"saved/image{page_index+1}_{image_index}.{image_ext}", "wb")) # сохранение в папке "saved"

