"""Everything in the outfit that depends on the playermodel.

With a male_08 dump (tools/gmod/dump_male08.lua) the values are measured and solved
on the real skeleton and mesh. Without one, Fit(None) falls back to the old outfit's
generic-skeleton values so the pipeline can run; that output is NOT fitted to male_08.
"""
import os

import numpy as np

import meshkit as mk
import oxygen_cart as oc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Fit:
    def __init__(self, dump=None):
        self.model = None
        self.fitted = False
        self._generic()
        if dump:
            import male08
            self.model = male08.Model(dump)
            import fit_male08
            fit_male08.apply(self, self.model)
            self.fitted = True

    def _generic(self):
        # posture (old outfit, tuned on a generic ValveBiped skeleton)
        self.pelvis_sink = (0, -0.8, 0)
        self.hunch = [(0, 3, 0), (0, 5, 0), (0, 9, 0), (0, 13, 0)]
        self.neck = (0, 36, 0)
        self.neck_walk = 5.0
        self.neck_aim_up = 0.0
        self.neck_aim_down = 0.0
        self.thigh, self.calf, self.foot = (0, -8, 0), (0, 16, 0), (0, -8, 0)
        self.clavicle_l, self.clavicle_r = (-8, 0, 0), (8, 0, 0)
        self.upperarm_hang, self.forearm_bend = (0, -28, 0), (0, -10, 0)

        self.cane_hand = ((2.79, -0.133, -0.031), (89.63, -89.45, 85.33))
        self.cane_arm = {"right upperarm": (1.76, -18.15, 7.02), "right forearm": (0, -32.78, 0),
                         "right hand": (-26.95, -0.25, 5.67)}
        self.cane_belt = ((-4.0, 2.0, -7.5), (0, 0, 0))

        self.cig_hand = ((4.785, -0.625, 0.763), (-33.2, 55.54, -172.3))
        self.cig_hand_tip = (6.1391, 1.3482, 2.329)
        self.cig_arm = {"left upperarm": ((13.01, 8.53, -25.27), (19.29, -78.45, 33.64)),
                        "left forearm": ((0, -88.22, 0), (0, -18.71, 0)),
                        "left hand": ((-38.3, -12.32, -13.34), (15.23, 14.97, -8.08))}
        self.mouth = (0.861, -4.865, 0)
        self.exhale_angles = (0, -90, 0)
        self.cig_mouth = ((0.9, -4.9, 1.0), (0, -100, 0))
        self.cig_mouth_tip = (0.9, -7.6, 1.6)

        self.glasses_obj = "oldman/oldman_glasses.obj"
        self.lenses_obj = "oldman/oldman_lenses.obj"

        self.mask_outlet = ((-0.9, -6.2, 0), (0, -90, 0))
        self.tube_clip_bone = "spine 2"
        self.tube_clip = ((4, 6, 2), (90, 0, 0))
        self.tube_bend_upper = 2.0
        self.tube_bend_lower = 9.0

        self.cart_pos = (0, 0, 0)
        self.cart_ang = (0, 0, 0)
        self.cart_speed = 8.0
        self.cart_strain = 0.45
        self.barb_pos = tuple(oc.barb_tip())
        self.wheels = [("left", (oc.WHEEL_X, oc.WHEEL_Y, oc.WHEEL_R)), ("right", (oc.WHEEL_X, -oc.WHEEL_Y, oc.WHEEL_R))]
        self.wheel_deg_per_unit = np.degrees(1.0 / oc.WHEEL_R)

    def neck_expression(self, walk):
        """Neck crane: more while walking (he leans further), less when he aims up and
        more when he aims down, so the head follows the aim instead of over-bending."""
        expr = f"{self.neck[1]:g} + {self.neck_walk:g}*{walk}"
        if self.neck_aim_up:
            expr += f" - {self.neck_aim_up:g}*clamp((owner_eye_angle_pitch() - 0.5)*2, 0, 1)"
        if self.neck_aim_down:
            expr += f" + {self.neck_aim_down:g}*clamp((0.5 - owner_eye_angle_pitch())*2, 0, 1)"
        return expr

    def write_assets(self, repo):
        props = os.path.join(repo, "props")
        os.makedirs(props, exist_ok=True)
        oc.paint_atlas(os.path.join(props, "o2_cart.jpg"))
        oc.cart_mesh().write_obj(os.path.join(props, "o2_cart.obj"),
                                 "portable oxygen cylinder on a cart - procedural, CC0\nunits: inches, origin on the ground under the cylinder")
        oc.wheel_mesh().write_obj(os.path.join(props, "o2_wheel.obj"),
                                  "oxygen cart wheel - procedural, CC0\ncentred on its axle, axle along y")
        if self.fitted:
            import fit_male08
            fit_male08.write_head_assets(self, props)
