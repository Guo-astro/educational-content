import os
from pathlib import Path

from manim import *

FLAGS = f"-v WARNING -pqm  --disable_caching "
SCENE = "FullTransformerAnimation"


from manim import *

class TransformerKeysAnimation(Scene):
    def construct(self):

        ############################################################
        # 1) INTRO / TITLE
        ############################################################
        title = Text(
            "Transformer: Calculating Keys",
            font_size=40, weight=BOLD
        )
        self.play(FadeIn(title, shift=UP))
        self.wait(1.5)
        self.play(title.animate.to_edge(UP))
        self.wait(0.5)

        ############################################################
        # Helper positions (x-coordinates for the 4 tokens)
        # We'll place them in a row: [-6, -2, +2, +6]
        ############################################################
        x_positions = [-6, -2, 2, 6]

        # We'll define vertical "levels" for clarity
        y_token   = -3  # Where the raw tokens go
        y_w_e     = -1  # W^E boxes
        y_plus    = 0   # plus sign for positional enc
        y_sum     = 1   # sum result
        y_w_k     = 2   # W^K boxes
        y_keys    = 3   # final k_i

        tokens = ["The", "car", "is", "blue"]

        ############################################################
        # 2) TOKEN EMBEDDINGS (512-d)
        ############################################################
        token_groups = VGroup()  # will hold 4 rectangles + labels
        dim_labels_512 = VGroup()  # "512" near each token

        for i, word in enumerate(tokens):
            # A rectangle to represent the token's embedding
            box = Rectangle(
                width=2.0, height=0.8,
                color=BLUE, stroke_width=2
            )
            txt = Text(word, font_size=24).move_to(box.get_center())
            group = VGroup(box, txt).move_to([x_positions[i], y_token, 0])
            token_groups.add(group)

            # Dimension label "512" to the right (or above)
            dim_label = Text("512", font_size=24, color=GREY_B)
            dim_label.next_to(group, RIGHT, buff=0.15)
            dim_labels_512.add(dim_label)

        # "embeddings" label on the left side
        embed_text = Text("embeddings", font_size=24)
        embed_text.to_edge(LEFT).shift(DOWN*1.0)

        self.play(
            LaggedStart(
                *[FadeIn(g, shift=DOWN) for g in token_groups],
                lag_ratio=0.3
            ),
            FadeIn(embed_text, shift=LEFT),
            run_time=2
        )
        self.play(*[FadeIn(lbl) for lbl in dim_labels_512])
        self.wait(1)

        ############################################################
        # 3) APPLY W^E? (In the GIF, "W^E" is sometimes shown.)
        #    We'll place a small W^E box above each token to emphasize
        #    that the raw word is turned into an embedding.
        #    (Alternatively, you can skip this if you only want the sum.)
        ############################################################
        w_e_boxes = VGroup()
        w_e_arrows = VGroup()
        for i in range(4):
            # W^E box
            w_e_rect = Rectangle(
                width=1.0, height=0.6,
                color=ORANGE, stroke_width=2
            )
            w_e_label = Tex(r"$W^E$", font_size=30).move_to(w_e_rect.get_center())
            w_e_group = VGroup(w_e_rect, w_e_label).move_to([x_positions[i], y_w_e, 0])

            # Arrow from token embedding to W^E
            arrow = Arrow(
                start=token_groups[i].get_top(),
                end=w_e_group.get_bottom(),
                buff=0.1
            )
            w_e_boxes.add(w_e_group)
            w_e_arrows.add(arrow)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in w_e_arrows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(box) for box in w_e_boxes], lag_ratio=0.2),
            run_time=2
        )
        self.wait(1)

        # (Optional) We'll keep the dimension 512 the same here,
        # so no dimension label changes.
        # But if you wanted to show "512->512", you could do that.

        ############################################################
        # 4) POSITIONAL ENCODING + PLUS SIGN
        ############################################################
        # We'll create a plus sign for each column, and a small
        # "positional encoding" label to the right of it.
        plus_signs = VGroup()
        posenc_arrows = VGroup()
        posenc_labels = VGroup()  # label "positional encodings"

        for i in range(4):
            plus = MathTex(r"+", font_size=50)
            plus.move_to([x_positions[i], y_plus, 0])

            # Arrow from W^E to plus sign
            arrow_we_to_plus = Arrow(
                start=w_e_boxes[i].get_top(),
                end=plus.get_bottom(),
                buff=0.1
            )

            # A small label to the right: "pos enc"
            pos_lbl = Text("pos enc", font_size=20, color=GREY_B)
            pos_lbl.next_to(plus, RIGHT, buff=0.3)

            plus_signs.add(plus)
            posenc_arrows.add(arrow_we_to_plus)
            posenc_labels.add(pos_lbl)

        # Also a big side label "positional encodings"
        posenc_text = Text("positional encodings", font_size=24)
        posenc_text.to_edge(RIGHT).shift(DOWN*1.0)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in posenc_arrows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(p) for p in plus_signs], lag_ratio=0.2),
            FadeIn(posenc_text, shift=RIGHT),
            run_time=2
        )
        self.play(*[FadeIn(lbl) for lbl in posenc_labels])
        self.wait(1)

        ############################################################
        # 5) SUM (embedding + pos enc) -> STILL 512-d
        #    We'll represent that sum as a rectangle labeled "512".
        ############################################################
        sum_boxes = VGroup()
        sum_arrows = VGroup()
        sum_dim_labels = VGroup()

        for i in range(4):
            sum_rect = Rectangle(
                width=1.6, height=0.8,
                color=YELLOW, stroke_width=2
            )
            # We'll label the sum as x_i with dimension 512
            sum_label = Tex(r"$x_{%d}$"%(i+1), font_size=30)
            sum_label.move_to(sum_rect.get_center())

            sum_group = VGroup(sum_rect, sum_label).move_to([x_positions[i], y_sum, 0])

            # arrow from plus sign to sum
            arrow_plus_to_sum = Arrow(
                start=plus_signs[i].get_top(),
                end=sum_group.get_bottom(),
                buff=0.1
            )

            # dimension label "512"
            dim_label = Text("512", font_size=24, color=GREY_B)
            dim_label.next_to(arrow_plus_to_sum, RIGHT, buff=0.05)

            sum_boxes.add(sum_group)
            sum_arrows.add(arrow_plus_to_sum)
            sum_dim_labels.add(dim_label)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in sum_arrows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(b) for b in sum_boxes], lag_ratio=0.2),
            run_time=2
        )
        self.play(*[FadeIn(dl) for dl in sum_dim_labels])
        self.wait(1)

        ############################################################
        # 6) W^K (512 -> 64), "calculating keys"
        ############################################################
        # We'll place a green box labeled W^K above each sum box.
        w_k_boxes = VGroup()
        w_k_arrows = VGroup()
        w_k_dim_labels = VGroup()

        for i in range(4):
            w_k_rect = Rectangle(
                width=1.2, height=0.6,
                color=GREEN, stroke_width=2
            )
            w_k_label = Tex(r"$W^K$", font_size=30).move_to(w_k_rect.get_center())
            w_k_group = VGroup(w_k_rect, w_k_label).move_to([x_positions[i], y_w_k, 0])

            arrow_sum_to_wk = Arrow(
                start=sum_boxes[i].get_top(),
                end=w_k_group.get_bottom(),
                buff=0.1
            )
            # dimension label "512 -> 64" near that arrow
            dim_label = Text("512 → 64", font_size=24, color=GREY_B)
            dim_label.next_to(arrow_sum_to_wk, RIGHT, buff=0.05)

            w_k_boxes.add(w_k_group)
            w_k_arrows.add(arrow_sum_to_wk)
            w_k_dim_labels.add(dim_label)

        # A side text "calculating keys"
        calc_keys_text = Text("calculating keys", font_size=24)
        calc_keys_text.to_edge(RIGHT).shift(UP*0.5)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in w_k_arrows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(wk) for wk in w_k_boxes], lag_ratio=0.2),
            FadeIn(calc_keys_text, shift=RIGHT),
            run_time=2
        )
        self.play(*[FadeIn(dl) for dl in w_k_dim_labels])
        self.wait(1)

        ############################################################
        # 7) FINAL KEYS (k_i), dimension 64
        ############################################################
        key_boxes = VGroup()
        key_arrows = VGroup()
        key_dim_labels = VGroup()

        for i in range(4):
            k_rect = Rectangle(
                width=1.2, height=0.6,
                color=GREEN, stroke_width=2
            )
            k_label = Tex(r"$k_{%d}$"%(i+1), font_size=24).move_to(k_rect.get_center())
            k_group = VGroup(k_rect, k_label).move_to([x_positions[i], y_keys, 0])

            arrow_wk_to_key = Arrow(
                start=w_k_boxes[i].get_top(),
                end=k_group.get_bottom(),
                buff=0.1
            )

            # dimension label "64"
            dim_label = Text("64", font_size=24, color=GREY_B)
            dim_label.next_to(arrow_wk_to_key, RIGHT, buff=0.05)

            key_boxes.add(k_group)
            key_arrows.add(arrow_wk_to_key)
            key_dim_labels.add(dim_label)

        self.play(
            LaggedStart(*[GrowArrow(a) for a in key_arrows], lag_ratio=0.2),
            LaggedStart(*[FadeIn(k) for k in key_boxes], lag_ratio=0.2),
            run_time=2
        )
        self.play(*[FadeIn(dl) for dl in key_dim_labels])
        self.wait(2)

        ############################################################
        # 8) CONCLUSION
        ############################################################
        concluding_text = Text(
            "Keys (k_i) computed!",
            font_size=28, color=YELLOW
        )
        concluding_text.to_edge(DOWN)

        self.play(Write(concluding_text))
        self.wait(2)

        # Fade out everything at the end
        self.play(
            FadeOut(title),
            FadeOut(embed_text),
            FadeOut(token_groups),
            FadeOut(dim_labels_512),
            FadeOut(w_e_boxes),
            FadeOut(w_e_arrows),
            FadeOut(plus_signs),
            FadeOut(posenc_arrows),
            FadeOut(posenc_labels),
            FadeOut(posenc_text),
            FadeOut(sum_boxes),
            FadeOut(sum_arrows),
            FadeOut(sum_dim_labels),
            FadeOut(w_k_boxes),
            FadeOut(w_k_arrows),
            FadeOut(w_k_dim_labels),
            FadeOut(key_boxes),
            FadeOut(key_arrows),
            FadeOut(key_dim_labels),
            FadeOut(calc_keys_text),
            FadeOut(concluding_text),
        )
        self.wait(1)


if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
    script_name = f"{Path(__file__).resolve()}"
    os.system(f"manim {script_name} {SCENE} {FLAGS}")
