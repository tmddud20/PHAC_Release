import argparse
import pprint
import yaml


CONTROL_CHANNELS = {
    "ib": 3,
    "eb": 3,
    "po": 3,
    "poib": 6,
    "poeb": 6,
}


def parse_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BBox/pose image-completion training")

    global_group = parser.add_argument_group("global")
    global_group.add_argument("--config", default="config.yaml")
    global_group.add_argument("--output_dir", default="../checkpoints")
    global_group.add_argument("--exp_name", default="image_completion")
    global_group.add_argument("--seed", type=int, default=None)
    global_group.add_argument(
        "--foundation_ckpt",
        default="../checkpoints/pix2gestalt.ckpt",
        help="Path to the Pix2Gestalt foundation checkpoint.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train_data_dir", required=False)
    dataset_group.add_argument("--validation_data_dir", default=None)
    dataset_group.add_argument("--dataloader_num_workers", type=int, default=0)
    dataset_group.add_argument("--resolution", type=int, default=512)
    dataset_group.add_argument(
        "--control_type",
        choices=tuple(CONTROL_CHANNELS),
        default="po",
        help="ib: interested bbox, eb: entire bbox, po: pose, "
        "poib: pose+interested bbox, poeb: pose+entire bbox",
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--train_batch_size", type=int, default=4)
    training_group.add_argument("--num_train_epochs", type=int, default=100)
    training_group.add_argument("--max_train_steps", type=int, default=None)
    training_group.add_argument("--checkpointing_steps", type=int, default=500)
    training_group.add_argument("--checkpoints_total_limit", type=int, default=None)
    training_group.add_argument("--resume_from_checkpoint", default=None)
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=1)
    training_group.add_argument("--gradient_checkpointing", action="store_true")

    validation_group = parser.add_argument_group("validation")
    validation_group.add_argument("--validation", action="store_true")
    validation_group.add_argument("--num_gen_images", type=int, default=4)
    validation_group.add_argument("--num_validation_images", type=int, default=4)
    validation_group.add_argument("--validation_steps", type=int, default=2500)

    network_group = parser.add_argument_group("network")
    network_group.add_argument("--drop_prob", type=float, default=0.1)
    network_group.add_argument("--unet_all", action="store_true")
    network_group.add_argument("--guidance_scale", type=float, default=3.5)
    network_group.add_argument("--conditioning_scale", type=float, default=1.0)

    optimizer_group = parser.add_argument_group("optimizer")
    optimizer_group.add_argument("--lr", type=float, default=5e-6)
    optimizer_group.add_argument("--lr_controlnet", type=float, default=5e-5)
    optimizer_group.add_argument("--scale_lr", action="store_true")
    optimizer_group.add_argument("--lr_scheduler", default="constant")
    optimizer_group.add_argument("--lr_warmup_steps", type=int, default=500)
    optimizer_group.add_argument("--lr_num_cycles", type=int, default=1)
    optimizer_group.add_argument("--lr_power", type=float, default=1.0)
    optimizer_group.add_argument("--adam_beta1", type=float, default=0.9)
    optimizer_group.add_argument("--adam_beta2", type=float, default=0.999)
    optimizer_group.add_argument("--adam_weight_decay", type=float, default=1e-5)
    optimizer_group.add_argument("--adam_epsilon", type=float, default=1e-8)
    optimizer_group.add_argument("--max_grad_norm", type=float, default=1.0)

    misc_group = parser.add_argument_group("misc")
    misc_group.add_argument("--report_to", default="wandb")
    misc_group.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true"
    )
    return parser


def parse_yaml_config(config_path: str, parser: argparse.ArgumentParser) -> None:
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    valid_fields = {action.dest for action in parser._actions}
    defaults = {}
    for group_name, fields in config.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Config group '{group_name}' must be a mapping")
        for field, value in fields.items():
            if field not in valid_fields:
                raise ValueError(f"Unknown config option: {field}")
            defaults[field] = value
    parser.set_defaults(**defaults)


def argparse_to_str(parser: argparse.ArgumentParser):
    initial_args, _ = parser.parse_known_args()
    if initial_args.config:
        parse_yaml_config(initial_args.config, parser)
    args = parser.parse_args()
    args.conditioning_channels = CONTROL_CHANNELS[args.control_type]

    grouped = {}
    for group in parser._action_groups:
        grouped[group.title] = {
            action.dest: getattr(args, action.dest, None)
            for action in group._group_actions
            if action.dest != "help"
        }
    grouped["network"]["conditioning_channels"] = args.conditioning_channels
    return args, pprint.pformat(grouped, indent=2)
