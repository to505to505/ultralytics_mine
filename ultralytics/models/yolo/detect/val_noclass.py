from ultralytics.utils.metrics import box_iou
import torch
from .val import DetectionValidator
from pathlib import Path


class NoClassDetectionValidator(DetectionValidator):
    """
    Аналог DetectionValidator, но без учёта классов.

    Логика:
      - Предполагаем, что batch["bboxes"] содержит список (или тензор) GT-боксов (форма может зависеть от вашего Dataset).
      - preds – это список (по числу картинок в batch), где каждая запись – тензор детекций [x1, y1, x2, y2, conf, cls].
      - Считаем TP/FP/FN/TN по принципу "есть ли вообще объекты в кадре".
    """


    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Переопределённая версия, игнорирующая классы в match_predictions.
        Вместо реальных классов просто передаём нули.
        """
        # iou для сопоставления боксов
        iou = box_iou(gt_bboxes, detections[:, :4])
        # Вместо detections[:, 5] и gt_cls передаем нули (один "общий" класс)
        return self.match_predictions(
            torch.zeros_like(detections[:, 5]),  # предсказанные классы обнуляем
            torch.zeros_like(gt_cls),            # gt-классы обнуляем
            iou
        )

    def update_metrics(self, preds, batch):
        """
        Переопределённая версия, где мы заменяем классы на нули
        перед вызовом _process_batch и обновлением confusion matrix.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)

            # Вместо реальных классов GT используем нули
            stat["target_cls"] = torch.zeros_like(cls)
            # Можно оставить cls.unique(), но для единообразия поставим нули
            stat["target_img"] = torch.zeros_like(cls)

            # Случай, когда нет предсказаний
            if npr == 0:
                if nl:
                    # Если GT есть, а предсказаний нет, добавляем статистику
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    # confusion_matrix тоже нужно вызвать (только если self.args.plots=True)
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None,
                            gt_bboxes=bbox,
                            gt_cls=stat["target_cls"]
                        )
                continue

            # Подготавливаем предсказания (NMS уже внутри postprocess)
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            # Аналогично — обнуляем классы у предсказанных объектов
            stat["pred_cls"] = torch.zeros_like(predn[:, 5])

            if nl:
                # Вычисляем tp, игнорируя классы
                # Для удобства передадим в _process_batch detections нужного вида (x1,y1,x2,y2,conf,cls)
                detections_for_match = torch.cat(
                    [predn[:, :4], predn[:, 4:5], stat["pred_cls"].unsqueeze(1)],
                    dim=1
                )
                stat["tp"] = self._process_batch(detections_for_match, bbox, stat["target_cls"])

                # Обновление confusion matrix (если нужно рисовать)
                if self.args.plots:
                    self.confusion_matrix.process_batch(
                        detections_for_match,
                        bbox,
                        stat["target_cls"]
                    )

            # Сохраняем накопленную статистику
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Сохранение в json/txt при необходимости
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )