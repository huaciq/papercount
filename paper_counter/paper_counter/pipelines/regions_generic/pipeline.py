"""Generic region interpretation and validation policy."""
from ...contracts import PipelineExecution
from ...shared.preparation import prepare_region_image,bright_core_bounds,attach_region_geometry


class GenericRegionsPipeline:
    pipeline_id='regions_generic_v1'

    def run(self,frame,recipe):
        cfg=recipe.to_counter_config()
        prepared=prepare_region_image(frame.image,cfg,stack_locator=bright_core_bounds)
        result=self.run_prepared(prepared.image,cfg)
        attach_region_geometry(result,prepared,frame.image,cfg)
        return PipelineExecution(result,prepared)

    def run_prepared(self,image,cfg):
        from ...shared.region_stages import run_region_stages
        return run_region_stages(image,cfg,lambda evidence:self.segment(evidence,cfg),self.semantic_warnings,self.validate)

    @staticmethod
    def validate(boxes,bands,observation,debug):
        from ...region_postvalidation import validate_box_sequence
        return validate_box_sequence(boxes,bands,observation,debug)

    @staticmethod
    def segment(e,cfg):
        from ...shared.region_stages import RegionDecision
        if e.observation is None:return RegionDecision(e.optical_units,e.structure,None)
        if cfg.region_structure=='auto':
            from ...region_sequence import repair_sequence
            units,audit=repair_sequence(e.optical_units,e.bands,e.observation,e.proposals,e.pitch,
                pair_evidence_score=e.pair_evidence_score,structure_hint=e.structure)
            structure=e.structure
            if audit.get('algorithm')=='supported_joint_resegmentation' and audit.get('scale_model'):
                e.identity['optical_structure_before_physical_sequence']=structure
                structure='paper_body'
        else:
            from ...region_thickness import repair_thickness
            units,audit=repair_thickness(e.optical_units,e.bands,e.observation,e.proposals,e.pitch)
            structure=e.structure
        return RegionDecision(units,structure,audit)

    @staticmethod
    def semantic_warnings(identity,selection,audit):
        warnings=[]
        if identity['ambiguous_pair_phase']:
            warnings.append('Pair phase is ambiguous; structure boxes do not prove physical paper identity.')
        if selection.get('physical_unit_ambiguous'):
            warnings.append('Physical unit cannot be uniquely identified from identical repeated bands; inspect or set region_structure explicitly.')
        return warnings
