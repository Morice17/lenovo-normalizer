"""
Canonical audience taxonomy for Lenovo Legion FY26 global media plans.

Aligned with the One-Vision Audience Console frontend (mockData.json).
Six segments, each with multiple "anchor" descriptions phrased the way
each region tends to write its targeting strings. The bi-encoder embeds
all anchors per segment and we match incoming raw strings against the
max similarity within each segment's anchor set.

This handles three structurally different input formats:
  - EMEA: short controlled-vocabulary labels
  - NA: narrative sentences
  - LATAM: long platform-export strings, often Spanish-language

Schema matches frontend expectations: id, label, definition, color.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class CanonicalSegment:
    id: str
    label: str
    definition: str
    color: str
    anchors: List[str] = field(default_factory=list)


TAXONOMY: List[CanonicalSegment] = [
    CanonicalSegment(
        id="immersed_gamer",
        label="Immersed Gamer",
        definition=(
            "High-intensity PC gamers seeking competitive performance, "
            "thermal headroom, and premium Legion hardware."
        ),
        color="#E2231A",
        anchors=[
            "Immersed PC Gamers",
            "Immersed Gamers",
            "18-24 Immersed Gamers, gaming 2+ hours per day, highly social and influenced by gaming communities, streamers, and esports",
            "Core PC gamers interested in Legion, high refresh displays, thermal performance",
            "18-34 high-intent PC gaming, premium GPUs, esports titles, Legion Pro consideration",
            "People Age 18-24, Interests: action games, multiplayer online battle arena, Call of Duty, massively multiplayer online games, League of Legends, esports",
            "Personas de 18 a 24 años, intereses: videojuegos de acción, esports, League of Legends, Call of Duty, jugadores intensivos",
            "Heavy PC gamers focused on premium hardware, frame rates, and immersive AAA titles",
        ],
    ),
    CanonicalSegment(
        id="performance_creator",
        label="Performance Creator",
        definition=(
            "Creator-gamers who use Legion devices for streaming, editing, "
            "and GPU-heavy production workflows."
        ),
        color="#F97316",
        anchors=[
            "Performance Creators",
            "Streamer creator audience comparing RTX laptops for editing and live play",
            "Twitch streamers and video editors needing one machine for work and ranked play",
            "Content creators and streamers using high-end GPUs for production and gaming",
            "People Age 22-40, Interests: Twitch streaming, video editing, Adobe Premiere, OBS, content creation, RTX laptops, NVIDIA Studio",
            "Creadores de contenido, streamers de Twitch, edicion de video, transmisiones en vivo, laptops para creadores",
            "Hybrid users who stream, edit video, and game competitively on the same hardware",
        ],
    ),
    CanonicalSegment(
        id="console_gamer",
        label="Console Gamer",
        definition=(
            "Console-first players evaluating PC gaming through ecosystem, "
            "value, and cross-platform compatibility."
        ),
        color="#8B5CF6",
        anchors=[
            "Console Gamers",
            "Console players considering PC Game Pass, affordable Legion entry models",
            "PlayStation and Xbox audiences, cross-platform Fortnite and FIFA players",
            "Console gamers evaluating PC for the first time, drawn by Game Pass and ecosystem",
            "People Age 18-34, Interests: PlayStation 5, Xbox Series X, Nintendo Switch, FIFA, Madden NFL, God of War, console gaming",
            "Jugadores de consola, PlayStation, Xbox, Nintendo Switch, juegos exclusivos de consola, FIFA",
            "Cross-platform players looking at PC as a complement to their console",
        ],
    ),
    CanonicalSegment(
        id="esports_aspirant",
        label="Esports Aspirant",
        definition=(
            "Competitive multiplayer audiences motivated by frame rates, "
            "refresh rates, and tournament culture."
        ),
        color="#00C896",
        anchors=[
            "Esports Aspirants",
            "Competitive multiplayer players browsing gaming laptops and performance benchmarks",
            "Competitive FPS fans, esports content viewers, laptop GPU upgrade intenders",
            "Ranked play enthusiasts focused on refresh rates, low latency, and tournament-grade hardware",
            "People Age 16-28, Interests: esports, competitive gaming, League of Legends World Championship, Counter-Strike, Valorant, professional gaming, gaming tournaments, ranked play",
            "Aspirantes a esports, jugadores competitivos, Counter-Strike, Valorant, League of Legends, torneos de gaming",
            "Players chasing competitive performance with high refresh, low input lag, tournament titles",
        ],
    ),
    CanonicalSegment(
        id="student_gamer",
        label="Student Gamer",
        definition=(
            "Students balancing study, social play, portability, and "
            "price-to-performance needs."
        ),
        color="#3B82F6",
        anchors=[
            "Student Gamers",
            "Gen Z and Student Gamers",
            "Students searching gaming laptop deals for school plus weekend gaming",
            "Gaming laptop for college, portability, battery life, student discounts",
            "Entry gaming laptop buyers, student gamers, value bundles",
            "People Age 16-22, Interests: university, college, student life, TikTok, study tips, Roblox, mobile games, K-pop",
            "Estudiantes universitarios, vida universitaria, laptops gaming economicas, descuentos estudiantiles, redes sociales",
            "University and high school students balancing study and casual gaming on a budget",
        ],
    ),
    CanonicalSegment(
        id="cloud_casual",
        label="Cloud Casual",
        definition=(
            "Light and cloud gaming audiences responding to accessibility, "
            "convenience, and subscription-led play."
        ),
        color="#F7B538",
        anchors=[
            "Cloud Casual",
            "Casual Gamers",
            "Cloud gaming and casual multiplayer, low-friction gaming access",
            "Light gamers using cloud streaming services like GeForce Now and Xbox Cloud Gaming",
            "People Age 25-44, Interests: puzzle games, casual games, Candy Crush, mobile games, GeForce Now, Xbox Cloud Gaming, subscription gaming",
            "Jugadores casuales, juegos en la nube, GeForce Now, juegos moviles, juegos por suscripcion, entretenimiento ocasional",
            "Casual and cloud-first players prioritizing convenience over hardware ownership",
        ],
    ),
]


# Confidence threshold below which we flag for human review.
# Per the proposal: per-class F1 >= 0.88 production gate, with a
# slightly wider confidence band for the free-text platform strings.
CONFIDENCE_THRESHOLD: float = 0.55

# Auto-approval threshold: above this, status = "auto_approved";
# between CONFIDENCE_THRESHOLD and AUTO_APPROVE_THRESHOLD, status = "needs_review".
# Below CONFIDENCE_THRESHOLD, predicted_label is None.
AUTO_APPROVE_THRESHOLD: float = 0.80


def get_labels() -> List[str]:
    return [seg.label for seg in TAXONOMY]


def get_ids() -> List[str]:
    return [seg.id for seg in TAXONOMY]


def get_anchor_map() -> Dict[str, List[str]]:
    """Returns {label: [anchor_text, ...]} for embedding."""
    return {seg.label: seg.anchors for seg in TAXONOMY}


def label_to_id(label: str) -> str:
    for seg in TAXONOMY:
        if seg.label == label:
            return seg.id
    raise KeyError(f"Unknown label: {label}")


def to_frontend_taxonomy() -> List[dict]:
    """Returns the taxonomy in the exact shape mockData.json expects."""
    return [
        {
            "id": seg.id,
            "label": seg.label,
            "definition": seg.definition,
            "color": seg.color,
        }
        for seg in TAXONOMY
    ]
