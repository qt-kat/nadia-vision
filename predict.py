import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Carga el modelo en GPU."""
        print("Iniciando setup: cargando el modelo...")
        model_id = "fancyfeast/llama-joycaption-beta-one-hf-llava"

        # Usar string "bfloat16" como en el ejemplo oficial
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="bfloat16",
            device_map=0
        )
        self.model.eval()
        print("Setup completado.")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Prompt for the model",
            default="Write a descriptive caption for this image."
        ),
        temperature: float = Input(
            description="Temperature for generation",
            default=0.6,
            ge=0.1,
            le=2.0
        ),
        max_new_tokens: int = Input(
            description="Maximum tokens to generate",
            default=512,
            ge=1,
            le=1024
        )
    ) -> str:
        """Ejecuta predicción sobre una imagen."""
        print("Procesando predicción...")

        with torch.no_grad():
            # Cargar imagen
            raw_image = Image.open(image)

            # Construir conversación
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]

            # Formatear con chat template
            convo_string = self.processor.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True
            )

            # Validación importante del ejemplo oficial
            assert isinstance(convo_string, str)

            # Procesar inputs
            inputs = self.processor(
                text=[convo_string],
                images=[raw_image],
                return_tensors="pt"
            ).to('cuda')

            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            # Generar con parámetros exactos del ejemplo oficial
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=temperature,
                top_k=None,
                top_p=0.9,
            )[0]

            # Recortar el prompt
            generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

            # Decodificar
            caption = self.processor.tokenizer.decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            caption = caption.strip()

            print(f"Caption generado: {caption}")
            return caption