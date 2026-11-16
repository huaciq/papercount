"""Provenance exports shared by compact/full and track/region outputs."""
from pathlib import Path
import yaml
import json


def save_recipe_metadata(result,directory):
    recipe=result.debug_info.get('resolved_recipe')
    if recipe is None:return
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'resolved_recipe.yaml').write_text(
        yaml.safe_dump(recipe,allow_unicode=True,sort_keys=False),encoding='utf-8')
    from .instances import instances_from_raw
    instances=instances_from_raw(result)
    (directory/'instances.json').write_text(json.dumps(dict(schema_version=1,count=result.count,
        count_unit=recipe['count_unit'],pipeline_id=recipe['pipeline'],recipe_id=recipe['recipe_id'],
        coordinate_system='original_input',instances=[item.to_dict() for item in instances]),
        ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
