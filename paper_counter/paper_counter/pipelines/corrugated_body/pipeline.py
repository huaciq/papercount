"""Corrugated body identity is decided here, outside the generic detector."""
from ...contracts import PipelineExecution
from ...shared.preparation import prepare_region_image,bright_core_bounds,attach_region_geometry


class CorrugatedBodyPipeline:
    pipeline_id='corrugated_body_v1'

    def run(self,frame,recipe):
        cfg=recipe.to_counter_config()
        prepared=prepare_region_image(frame.image,cfg,stack_locator=bright_core_bounds)
        result=self.run_prepared(prepared.image,cfg)
        attach_region_geometry(result,prepared,frame.image,cfg)
        return PipelineExecution(result,prepared)

    def run_prepared(self,image,cfg):
        from ...shared.region_stages import run_region_stages
        return run_region_stages(image,cfg,self.segment,self.semantic_warnings,self.validate)

    @staticmethod
    def validate(boxes,bands,observation,debug):
        from ...region_postvalidation import validate_box_sequence
        return validate_box_sequence(boxes,bands,observation,debug)

    @staticmethod
    def segment(e):
        from ...shared.region_stages import RegionDecision
        from ...corrugated_regions import segment_corrugated_bodies
        units,audit=segment_corrugated_bodies(e.aligned_raw,e.valid,e.bands,e.optical_units,e.pitch)
        e.identity['optical_structure_before_physical_sequence']=e.structure
        return RegionDecision(units,'paper_body',audit)

    @staticmethod
    def semantic_warnings(identity,selection,audit):
        unresolved=(audit or {}).get('unresolved_contact_regions',[])
        return ([f'{len(unresolved)} wide corrugated bodies lack another observed seam; touching sheets may remain merged.']
                if unresolved else [])
